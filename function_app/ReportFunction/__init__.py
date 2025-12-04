import os
import json
import requests
import azure.functions as func
from typing import Any, Optional

# =====================================================
# 🔧 ENVIRONMENT VARIABLES
# =====================================================
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4-fast-reasoning")

TIMEOUT_GROK = 60


# =====================================================
# 🔧 UTILIDADES
# =====================================================
def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except:
        return resp.text


# =====================================================
# 🔧 GROK MEDICAL PLAN GENERATOR (WITH GRADCAM CONTEXT)
# =====================================================
def generate_medical_plan(
    lesion_type: str,
    lesion_probability: float,
    lesion_type_probability: float,
    patient_info: Optional[dict] = None,
    has_gradcam: bool = False
) -> dict:
    """
    Generates a comprehensive medical description and treatment plan
    using Grok based on diagnostic results.
    
    Args:
        lesion_type: The diagnosed lesion type (e.g., "melanoma", "basal_cell_carcinoma")
        lesion_probability: Probability that a lesion exists (0-1)
        lesion_type_probability: Confidence in the lesion type classification (0-1)
        patient_info: Optional dict with patient details (age, sex, medical_history)
        has_gradcam: Whether GradCAM heatmap visualization was generated
    
    Returns:
        dict with: description, treatment_plan, recommendations, urgency_level
    """
    if not XAI_API_KEY:
        return {
            "error": "XAI_API_KEY not configured",
            "description": "Unable to generate medical plan",
            "treatment_plan": "N/A",
            "recommendations": [],
            "urgency_level": "unknown",
            "follow_up": "N/A",
            "differential_diagnosis": []
        }

    # Build context from patient info
    patient_context = ""
    if patient_info:
        age = patient_info.get("age", "unknown")
        sex = patient_info.get("sex", "unknown")
        history = patient_info.get("medical_history", "none reported")
        patient_context = f"\nPatient: {age} years old, {sex}, Medical history: {history}"

    # Add GradCAM context if available
    gradcam_context = ""
    if has_gradcam:
        gradcam_context = "\n\nVISUAL ANALYSIS:\n- GradCAM heatmap generated showing regions of interest\n- The AI model focused on specific areas when making this classification\n- Visual attention patterns have been analyzed and are available for review"

    # Create detailed prompt for Grok
    user_prompt = f"""You are a dermatologist reviewing an AI-assisted skin lesion analysis with explainable AI visualization.

DIAGNOSTIC RESULTS:
- Lesion detected: {lesion_probability:.1%} probability
- Diagnosed type: {lesion_type if lesion_type != "no_lesion_detected" else "No lesion detected"}
- Classification confidence: {lesion_type_probability:.1%}
{patient_context}{gradcam_context}

Please provide a comprehensive medical assessment in JSON format with the following structure:
{{
  "description": "Detailed medical description of the condition (2-3 sentences)",
  "treatment_plan": "Recommended treatment approach (2-3 sentences)",
  "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
  "urgency_level": "low|moderate|high|urgent",
  "follow_up": "Follow-up timeline and monitoring recommendations",
  "differential_diagnosis": ["possible alternative diagnosis 1", "possible alternative diagnosis 2"]
}}

Guidelines:
- Be professional and evidence-based
- Use clear, patient-friendly language
- Always emphasize the need for professional medical evaluation
- Consider the confidence levels in your recommendations
- Base urgency on the lesion type and probabilities
{f"- Note that visual analysis (GradCAM) is available for clinical review" if has_gradcam else ""}
- Provide actionable, specific recommendations
- The description should explain what the condition is in accessible terms
- Treatment plan should outline next steps without being overly prescriptive
- Recommendations should be practical and prioritized
{"- If no lesion was detected (low probability), provide reassurance but still recommend monitoring" if lesion_type == "no_lesion_detected" else ""}

CRITICAL: Respond ONLY with valid JSON. No markdown, no preamble, no backticks."""

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": XAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert dermatologist providing medical assessments with access to "
                    "AI-assisted diagnostic tools including explainable AI visualizations. "
                    "Always provide cautious, evidence-based recommendations. "
                    "Respond ONLY in valid JSON format without any markdown formatting. "
                    "When GradCAM visualization is available, acknowledge its value for clinical correlation "
                    "but maintain focus on evidence-based clinical assessment."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,  # Lower temperature for more consistent medical advice
    }

    try:
        resp = requests.post(
            f"{XAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=TIMEOUT_GROK,
        )
        resp.raise_for_status()
        
        data = _safe_json(resp)
        
        # Extract the response content
        content = data["choices"][0]["message"]["content"]
        
        # Clean up any markdown formatting
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON response
        medical_plan = json.loads(content)
        
        # Ensure all required fields are present
        required_fields = {
            "description": "Medical assessment unavailable",
            "treatment_plan": "Please consult a dermatologist",
            "recommendations": ["Seek professional medical evaluation"],
            "urgency_level": "moderate",
            "follow_up": "Schedule appointment with dermatologist",
            "differential_diagnosis": []
        }
        
        for field, default_value in required_fields.items():
            if field not in medical_plan:
                medical_plan[field] = default_value
        
        return medical_plan
        
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parsing Grok response as JSON: {e}")
        print(f"Raw content: {content if 'content' in locals() else 'N/A'}")
        return {
            "error": "Failed to parse response",
            "description": "The AI analysis suggests further evaluation is needed. This appears to be a skin lesion that warrants professional dermatological examination.",
            "treatment_plan": "Schedule an appointment with a board-certified dermatologist for clinical evaluation and potential biopsy if indicated.",
            "recommendations": [
                "Seek professional dermatological evaluation",
                "Document any changes in size, color, or symptoms",
                "Avoid self-treatment without medical guidance"
            ],
            "urgency_level": "moderate",
            "follow_up": "Consult dermatologist within 2-4 weeks",
            "differential_diagnosis": []
        }
    except Exception as e:
        print(f"❌ Error calling Grok API: {e}")
        return {
            "error": str(e),
            "description": "Unable to generate detailed medical plan. Professional evaluation is strongly recommended.",
            "treatment_plan": "Please consult a board-certified dermatologist for proper diagnosis and treatment.",
            "recommendations": [
                "Seek immediate professional medical evaluation",
                "Do not delay consultation",
                "Bring any relevant medical history"
            ],
            "urgency_level": "moderate",
            "follow_up": "Consult dermatologist as soon as possible",
            "differential_diagnosis": []
        }


# =====================================================
# 🔥 AZURE FUNCTION HTTP ENTRYPOINT
# =====================================================
def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function endpoint for generating medical plans.
    
    Expected JSON input:
    {
        "lesion_type": "melanoma",
        "lesion_probability": 0.85,
        "lesion_type_probability": 0.92,
        "has_gradcam": true,
        "patient_info": {
            "age": 45,
            "sex": "Male",
            "medical_history": "No prior skin conditions"
        }
    }
    """
    try:
        # Parse request body
        req_body = req.get_json()
        
        lesion_type = req_body.get("lesion_type")
        lesion_probability = req_body.get("lesion_probability")
        lesion_type_probability = req_body.get("lesion_type_probability")
        patient_info = req_body.get("patient_info")
        has_gradcam = req_body.get("has_gradcam", False)
        
        # Validate required fields
        # lesion_type can be None if probability is low
        if lesion_probability is None:
            return func.HttpResponse(
                json.dumps({
                    "error": "Missing required field: lesion_probability"
                }),
                mimetype="application/json",
                status_code=400,
            )
        
        # If no lesion detected (null lesion_type), use a placeholder
        if not lesion_type:
            lesion_type = "no_lesion_detected"
            lesion_type_probability = 0.0
            print(f"✔️ No lesion detected (probability: {lesion_probability:.2%})")
        else:
            print(f"✔️ Generating medical plan for: {lesion_type} ({lesion_probability:.2%})")
        if has_gradcam:
            print(f"   📊 GradCAM visualization available")
        
        # Generate medical plan
        medical_plan = generate_medical_plan(
            lesion_type=lesion_type,
            lesion_probability=lesion_probability,
            lesion_type_probability=lesion_type_probability or 0.0,
            patient_info=patient_info,
            has_gradcam=has_gradcam
        )
        
        # Add input data to response for reference
        response = {
            "input": {
                "lesion_type": lesion_type,
                "lesion_probability": lesion_probability,
                "lesion_type_probability": lesion_type_probability,
                "has_gradcam": has_gradcam
            },
            "medical_plan": medical_plan
        }
        
        return func.HttpResponse(
            json.dumps(response, ensure_ascii=False, indent=2),
            mimetype="application/json",
            status_code=200,
        )
        
    except ValueError as e:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON in request body"}),
            mimetype="application/json",
            status_code=400,
        )
    except Exception as e:
        print(f"❌ ERROR GENERAL: {e}")
        import traceback
        traceback.print_exc()
        
        return func.HttpResponse(
            json.dumps({"error": f"Internal error: {str(e)}"}),
            mimetype="application/json",
            status_code=500,
        )