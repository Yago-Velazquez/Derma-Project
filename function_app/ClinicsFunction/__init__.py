import json
import os
import azure.functions as func
import requests

# ================
# ENV VARIABLES
# ================
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4-fast-reasoning")

TIMEOUT_GROK = 40


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        city = body.get("city")

        if not city:
            return func.HttpResponse(
                "Missing field: 'city'",
                status_code=400
            )

        # Ask Grok for structured clinics with logo URLs
        system_prompt = (
            "You are a medical assistant that provides a list of the best "
            "dermatology clinics in a given city. "
            "You ALWAYS reply in strict JSON with fields: "
            "[name, address, phone, website, lat, lon, logo_url]. "
            "For logo_url: provide a direct link to the clinic's logo image if available. "
            "Search their website or use common logo sources. If unavailable, set to empty string. "
            "Prefer PNG or transparent logos when possible. "
            "If data is unavailable, estimate approximate coordinates of the city center "
            "and leave missing fields empty."
        )

        user_prompt = (
            f"Give me the 5 best dermatology clinics in {city}. "
            "Include their logo URLs if you can find them. "
            "Return strict JSON. No text outside JSON."
        )

        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": XAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

        response = requests.post(
            f"{XAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=TIMEOUT_GROK
        )

        response.raise_for_status()
        data = response.json()

        # Should be pure JSON text in message
        content = data["choices"][0]["message"]["content"]

        try:
            clinics = json.loads(content)
        except:
            # fallback: deliver raw text
            clinics = {"raw": content}

        result = {
            "city": city,
            "clinics": clinics
        }

        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        return func.HttpResponse(
            f"ClinicsFunction error: {str(e)}",
            status_code=500
        )