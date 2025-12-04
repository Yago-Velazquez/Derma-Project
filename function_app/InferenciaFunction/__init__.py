import os
import json
import base64
import requests
import azure.functions as func
from typing import Any, Optional, Tuple
from PIL import Image
import io


# =====================================================
# 🔧 PREPROCESADO PARA MODELO 2
# =====================================================
def preprocess_image_for_model2(image_bytes: bytes) -> bytes:
    """
    Preprocesses image for Model 2:
    - Converts HEIF if needed
    - Crops to square
    - Resizes to 224x224
    """
    try:
        # Try loading as standard image
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            # If fails, try HEIF
            try:
                from pillow_heif import read_heif
                heif = read_heif(io.BytesIO(image_bytes))
                img = Image.frombytes(heif.mode, heif.size, heif.data)
            except ImportError:
                print("⚠️ pillow-heif not installed, cannot process HEIF images")
                return image_bytes
            except Exception as e:
                print(f"⚠️ Error processing HEIF: {e}")
                return image_bytes

        img = img.convert("RGB")
        """"
        # Crop to square (center crop)
        w, h = img.size
        m = min(w, h)
        left = (w - m) // 2
        top = (h - m) // 2
        img = img.crop((left, top, left + m, top + m))

        # Resize to 224x224
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        """
        # Convert back to bytes
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    except Exception as e:
        print(f"⚠️ Error preprocessing image: {e}")
        return image_bytes


# =====================================================
# 🔧 UTILIDADES
# =====================================================
def _to_b64(b: bytes) -> str:
    """Convert bytes to base64 string."""
    return base64.b64encode(b).decode("utf-8")


def _post_json(url: str, token: str, payload: dict, timeout: int) -> requests.Response:
    """Make authenticated POST request with JSON payload."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    return requests.post(url, headers=headers, json=payload, timeout=timeout)


def _safe_json(resp: requests.Response) -> Any:
    """Safely parse JSON response, fall back to text."""
    try:
        return resp.json()
    except:
        return resp.text


def _parse_probability(data: Any) -> float:
    """Extract probability from various response formats."""
    if isinstance(data, dict):
        # Try common keys
        for k in ("probability", "prob", "score", "prediction", "pred"):
            if k in data:
                return float(data[k])

    if isinstance(data, list) and data:
        return float(data[0])

    if isinstance(data, (float, int, str)):
        return float(data)

    raise RuntimeError(f"Formato de probabilidad no reconocido: {data}")


def _parse_label(data: Any) -> Optional[str]:
    """Extract label from various response formats."""
    if isinstance(data, dict):
        if "label" in data:
            return data["label"]

        if "labels" in data and isinstance(data["labels"], list):
            return data["labels"][0]

        # Recursive search in nested structures
        for key in ("result", "output", "data"):
            if key in data:
                return _parse_label(data[key])

    if isinstance(data, list) and data:
        if isinstance(data[0], dict) and "label" in data[0]:
            return data[0]["label"]
        if isinstance(data[0], str):
            return data[0]

    if isinstance(data, str):
        return data

    return None


# =====================================================
# 🔧 ENVIRONMENT VARIABLES
# =====================================================
MODEL2_URL = os.getenv("MODEL2_URL")
MODEL2_TOKEN = os.getenv("MODEL2_TOKEN")
THRESHOLD = float(os.getenv("THRESHOLD", "0.7"))

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4-fast-reasoning")

TIMEOUT_GROK = int(os.getenv("TIMEOUT_GROK", "40"))
TIMEOUT_MODEL2 = int(os.getenv("TIMEOUT_MODEL2", "40"))


# =====================================================
# 🔧 GROK VISION - SKIN LESION DETECTION
# =====================================================
def call_grok_lesion_detection(image_bytes: bytes) -> Tuple[bool, float]:
    """
    Call Grok Vision API to detect if image contains a skin lesion.
    
    Returns:
        (is_lesion, confidence)
    """
    if not XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY no configurado")

    image_b64 = _to_b64(image_bytes)

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
                    "You are an image classifier specialized in detecting whether an image "
                    "contains a human skin lesion (mole, nevus, melanoma, rash, wound, scar, etc.). "
                    "Your ONLY task is to determine:\n"
                    " - Yes  → if the image shows a human skin lesion\n"
                    " - No   → if it does NOT\n\n"
                    "Rules:\n"
                    " - Do NOT provide medical advice.\n"
                    " - Do NOT identify lesion types.\n"
                    " - Output ONLY the following format:\n"
                    "   Yes/No <space> <confidence between 0 and 1>\n"
                    "Example: Yes 0.84\n"
                    "If the image is ambiguous, choose the best option and lower confidence."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Does this image show a skin lesion? "
                            "Respond ONLY with: Yes/No and confidence."
                        )
                    }
                ]
            }
        ],
        "temperature": 0.1,
    }


    resp = requests.post(
        f"{XAI_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=TIMEOUT_GROK,
    )

    resp.raise_for_status()
    data = _safe_json(resp)

    content = data["choices"][0]["message"]["content"].strip()
    parts = content.split()
    answer = parts[0].lower()
    confidence = float(parts[1]) if len(parts) > 1 else 0.5

    is_lesion = answer == "yes"
    prob = confidence if is_lesion else (1 - confidence)

    return is_lesion, prob


# =====================================================
# 🔧 MODEL 2 CALL (with GradCAM)
# =====================================================
def call_model2(image_bytes: bytes) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Call Azure ML Model 2 endpoint for lesion classification.
    
    Returns:
        (label, probability, gradcam_b64)
    """
    if not MODEL2_URL or not MODEL2_TOKEN:
        raise RuntimeError("MODEL2_URL / MODEL2_TOKEN no configurados")

    payload = {
        "image": _to_b64(image_bytes),
        "include_gradcam": True  # Request GradCAM from endpoint
    }
    
    resp = _post_json(MODEL2_URL, MODEL2_TOKEN, payload, TIMEOUT_MODEL2)
    resp.raise_for_status()

    data = _safe_json(resp)

    # If response is string, try to parse as JSON
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            pass

    label = _parse_label(data)
    probability = _parse_probability(data)
    gradcam_b64 = data.get("gradcam", None) if isinstance(data, dict) else None

    return label, probability, gradcam_b64


# =====================================================
# 🔧 GROK EXPLANATION
# =====================================================
def call_grok_explanation(
    lesion_type: Optional[str], 
    lesion_type_probability: Optional[float], 
    prob: float
) -> str:
    """
    Generate AI explanation of the analysis results using Grok.
    """
    if not XAI_API_KEY:
        return "No se ha podido generar explicación automática."

    if lesion_type:
        user_msg = (
            f"Skin image analysis results:\n"
            f"- Lesion detected with probability: {prob:.2%}\n"
            f"- Estimated lesion type: {lesion_type} (confidence: {lesion_type_probability:.2%})\n\n"
            "Write a brief, cautious medical explanation in English. "
            "Emphasize this is not a diagnosis and recommend professional consultation."
        )
    else:
        user_msg = (
            f"Skin image analysis results:\n"
            f"- Low probability of lesion detected: {prob:.2%}\n\n"
            "Write a brief explanation in English noting the low probability, "
            "but still recommend professional evaluation if concerned."
        )

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
                    "You are a medical AI assistant helping explain skin lesion analysis results. "
                    "Be clear, cautious, and empathetic. Never provide definitive diagnoses. "
                    "Always recommend professional medical consultation. Keep responses concise (2-3 paragraphs)."
                )
            },
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
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
        return data["choices"][0]["message"]["content"]
    
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return "Unable to generate automatic explanation. Please consult with a healthcare professional."


# =====================================================
# 🔥 ENTRYPOINT HTTP
# =====================================================
def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function HTTP trigger.
    
    Orchestrates the full analysis pipeline:
    1. Grok Vision: Detect if lesion present
    2. Model 2: Classify lesion type (if detected)
    3. Grok: Generate explanation
    
    Returns JSON with results.
    """
    try:
        # Get image from request body
        raw = req.get_body()

        if not raw:
            return func.HttpResponse(
                json.dumps({"error": "No image received in request body"}),
                mimetype="application/json",
                status_code=400
            )

        print(f"✔️ Image received, size: {len(raw)} bytes")

        # Step 1: Grok Vision - Lesion Detection
        print("Step 1: Calling Grok Vision for lesion detection...")
        is_lesion, prob = call_grok_lesion_detection(raw)
        print(f"  Result: is_lesion={is_lesion}, probability={prob:.2f}")

        # Step 2: If likely lesion → classify type with Model 2
        lesion_type = None
        lesion_prob = 0.0
        gradcam_b64 = None

        if is_lesion and prob >= THRESHOLD:
            print(f"Step 2: Lesion detected (prob={prob:.2f} >= {THRESHOLD}), calling Model 2...")
            model_img = preprocess_image_for_model2(raw)
            lesion_type, lesion_prob, gradcam_b64 = call_model2(model_img)
            print(f"  Result: type={lesion_type}, probability={lesion_prob:.2f}")
        else:
            print(f"Step 2: Skipped (prob={prob:.2f} < {THRESHOLD} or not lesion)")

        # Step 3: AI-generated explanation
        print("Step 3: Generating explanation with Grok...")
        explanation = call_grok_explanation(lesion_type, lesion_prob, prob)

        # Build response
        result = {
            "is_lesion": is_lesion,
            "probability": prob,
            "lesion_type": lesion_type,
            "lesion_type_probability": lesion_prob,
            "explanation": explanation,
            "gradcam": gradcam_b64
        }

        print("✔️ Analysis complete")
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            mimetype="application/json",
            status_code=200,
        )

    except requests.HTTPError as e:
        error_msg = f"External API error: {e.response.status_code}"
        if e.response.text:
            error_msg += f" - {e.response.text[:200]}"
        
        print(f"❌ HTTP Error: {error_msg}")
        return func.HttpResponse(
            json.dumps({"error": error_msg}),
            mimetype="application/json",
            status_code=502,
        )

    except Exception as e:
        print(f"❌ GENERAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )