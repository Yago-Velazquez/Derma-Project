import os
import json
import requests
import azure.functions as func
from typing import Any, Optional, List, Dict

# =====================================================
# 🔧 ENVIRONMENT VARIABLES
# =====================================================
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://router.huggingface.co/google/medgemma-4b-it"


TIMEOUT = 60

# =====================================================
# 🔧 UTILIDADES
# =====================================================
def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except:
        return resp.text


# =====================================================
# 🔧 MEDGEMMA CHATBOT
# =====================================================
def call_medgemma_chat(
    user_message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = 500,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Calls MedGemma 4B IT model from Hugging Face for medical consultation.
    
    Args:
        user_message: The user's current message/question
        conversation_history: Optional list of previous messages [{"role": "user/assistant", "content": "..."}]
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0 to 1.0)
    
    Returns:
        dict with: response, conversation_history, error (if any)
    """
    if not HF_API_KEY:
        return {
            "error": "HF_API_KEY not configured",
            "response": "Unable to connect to medical assistant",
        }

    # Build conversation context
    if conversation_history is None:
        conversation_history = []
    
    # System prompt for medical context
    system_prompt = """You are MedGemma, a helpful and knowledgeable medical assistant. Your role is to:
- Provide accurate, evidence-based medical information
- Answer questions about symptoms, conditions, and treatments
- Always emphasize that you are not a replacement for professional medical advice
- Recommend consulting healthcare professionals for diagnosis and treatment
- Be empathetic and clear in your explanations
- Ask clarifying questions when needed

Remember: Always remind users to seek professional medical care for serious concerns."""

    # Format the conversation for MedGemma
    prompt = f"{system_prompt}\n\n"
    
    # Add conversation history
    for msg in conversation_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            prompt += f"Patient: {content}\n"
        elif role == "assistant":
            prompt += f"MedGemma: {content}\n"
    
    # Add current user message
    prompt += f"Patient: {user_message}\nMedGemma:"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
        }
    }

    try:
        print(f"✔️ Calling MedGemma with message: {user_message[:50]}...")
        
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        
        data = _safe_json(response)
        
        # Extract generated text
        if isinstance(data, list) and len(data) > 0:
            generated_text = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            generated_text = data.get("generated_text", str(data))
        else:
            generated_text = str(data)
        
        # Clean up the response
        generated_text = generated_text.strip()
        
        # Update conversation history
        updated_history = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": generated_text}
        ]
        
        print(f"✔️ MedGemma response generated: {len(generated_text)} chars")
        
        return {
            "response": generated_text,
            "conversation_history": updated_history,
            "tokens_used": len(generated_text.split())
        }
        
    except requests.HTTPError as e:
        error_msg = f"Hugging Face API error: {e.response.status_code}"
        try:
            error_detail = e.response.json()
            error_msg += f" - {error_detail}"
        except:
            error_msg += f" - {e.response.text}"
        
        print(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "response": "I'm having trouble connecting right now. Please try again in a moment.",
        }
    
    except Exception as e:
        print(f"❌ Error calling MedGemma: {e}")
        return {
            "error": str(e),
            "response": "An unexpected error occurred. Please try again.",
        }


# =====================================================
# 🔥 AZURE FUNCTION HTTP ENTRYPOINT
# =====================================================
def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function endpoint for MedGemma medical chatbot.
    
    Expected JSON input:
    {
        "message": "What are the symptoms of diabetes?",
        "conversation_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    """
    try:
        # Parse request body
        req_body = req.get_json()
        
        user_message = req_body.get("message")
        conversation_history = req_body.get("conversation_history", [])
        max_tokens = req_body.get("max_tokens", 500)
        temperature = req_body.get("temperature", 0.7)
        
        # Validate required fields
        if not user_message:
            return func.HttpResponse(
                json.dumps({
                    "error": "Missing required field: message"
                }),
                mimetype="application/json",
                status_code=400,
            )
        
        # Validate conversation history format
        if not isinstance(conversation_history, list):
            conversation_history = []
        
        print(f"✔️ Received message: {user_message}")
        
        # Call MedGemma
        result = call_medgemma_chat(
            user_message=user_message,
            conversation_history=conversation_history,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Check for errors
        if "error" in result and "response" in result:
            # Partial error - model responded but there was an issue
            return func.HttpResponse(
                json.dumps(result, ensure_ascii=False, indent=2),
                mimetype="application/json",
                status_code=200,  # Still return 200 with error info
            )
        
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False, indent=2),
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
        return func.HttpResponse(
            json.dumps({"error": f"Internal error: {str(e)}"}),
            mimetype="application/json",
            status_code=500,
        )