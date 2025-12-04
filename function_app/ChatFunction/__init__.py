import json
import os
import azure.functions as func
import requests
from typing import List, Dict, Any

# ================================
# 🔧 ENV VARS FOR GROK
# ================================
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4-fast-reasoning")

TIMEOUT_GROK = 40
MAX_HISTORY_TOKENS = 2000  # Limit total history tokens
MAX_HISTORY_MESSAGES = 10  # Limit number of previous messages

SYSTEM_PROMPT = (
    "You are a helpful dermatology assistant specializing in skin lesion analysis. "
    "Provide clear, empathetic, and scientifically grounded responses. "
    "Your goal is to help users understand their medical imaging results "
    "while always emphasizing the importance of professional medical consultation."
)

def truncate_history(history: List[Dict[str, str]], max_tokens: int = MAX_HISTORY_TOKENS) -> List[Dict[str, str]]:
    """
    Truncate conversation history to fit within token limit.
    Prioritizes most recent messages.
    """
    truncated_history = []
    current_tokens = 0

    # Reverse the history to process most recent messages first
    for message in reversed(history):
        # Estimate token count (rough approximation)
        message_tokens = len(message.get('content', '').split())
        
        if current_tokens + message_tokens <= max_tokens:
            truncated_history.insert(0, message)
            current_tokens += message_tokens
        else:
            break

    return truncated_history[-MAX_HISTORY_MESSAGES:]

def sanitize_context(context: Dict[str, Any]) -> str:
    """
    Create a sanitized context description from analysis results.
    """
    context_parts = []
    
    if context.get('probability'):
        context_parts.append(f"Lesion probability: {context['probability']}")
    
    if context.get('lesion_type'):
        context_parts.append(f"Estimated lesion type: {context['lesion_type']}")
    
    if context.get('lesion_type_probability'):
        context_parts.append(f"Lesion type confidence: {context['lesion_type_probability']}")
    
    if context.get('has_gradcam'):
        context_parts.append("Visual analysis: GradCAM heatmap available")
    
    return "Medical Imaging Context:\n" + "\n".join(context_parts) if context_parts else ""

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()

        # Validate required fields
        user_message = body.get("message")
        context = body.get("context", {})
        history = body.get("history", [])

        if not user_message:
            return func.HttpResponse(
                json.dumps({"error": "Missing field: 'message'"}),
                mimetype="application/json",
                status_code=400
            )

        # Truncate and sanitize conversation history
        sanitized_history = truncate_history(history)
        
        # Prepare context description
        context_description = sanitize_context(context)

        # Construct messages for API call
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context_description},
        ]

        # Add sanitized conversation history
        messages.extend([
            {"role": m.get("role"), "content": m.get("content")} 
            for m in sanitized_history 
            if m.get("role") in ["user", "assistant"] and m.get("content")
        ])

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": XAI_MODEL,
            "messages": messages,
            "temperature": 0.7,  # Slightly increased for more dynamic responses
            "max_tokens": 300,   # Limit response length
        }

        try:
            response = requests.post(
                f"{XAI_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT_GROK
            )

            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]

            return func.HttpResponse(
                json.dumps({"answer": answer}, ensure_ascii=False),
                mimetype="application/json",
                status_code=200
            )

        except requests.exceptions.Timeout:
            return func.HttpResponse(
                json.dumps({"error": "Request timed out. Please try again."}),
                mimetype="application/json",
                status_code=504
            )

        except requests.exceptions.RequestException as e:
            return func.HttpResponse(
                json.dumps({"error": f"API request failed: {str(e)}"}),
                mimetype="application/json",
                status_code=502
            )

    except Exception as e:
        print(f"❌ ChatFunction error: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Unexpected error: {str(e)}"}),
            mimetype="application/json",
            status_code=500
        )