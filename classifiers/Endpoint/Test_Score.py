import requests
import base64
import json
from PIL import Image
import io

# =====================================================
# CONFIGURATION
# =====================================================
ENDPOINT_URL = "https://end-tol.spaincentral.inference.ml.azure.com/score"
API_KEY = "5rlx8oeUZhnSuo5LCI5h9cx4sEGUNLr1R53K8Zl7qn5KcEPwAyF1JQQJ99BKAAAAAAAAAAAAINFRAZML2suo"
IMAGE_PATH = "/Users/yago.velazquez/Derma-Project/classifiers/modelo-2/sample.jpg"


def load_image_as_base64(image_path):
    """Load image and convert to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_endpoint_basic(endpoint_url, api_key, image_b64):
    """Test endpoint without GradCAM."""
    print("\n" + "="*60)
    print("TEST 1: Basic Prediction (no GradCAM)")
    print("="*60)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "image": image_b64
    }
    
    response = requests.post(endpoint_url, headers=headers, json=payload)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✔️ Success!")
        print(f"Predicted Label: {result.get('label')}")
        print(f"Confidence: {result.get('score', 0):.2%}")
        
        if 'all_scores' in result:
            print(f"\nTop 3 Predictions:")
            for i, item in enumerate(result['all_scores'][:3], 1):
                print(f"  {i}. {item['label']}: {item['score']:.2%}")
        
        return result
    else:
        print(f"❌ Error: {response.text}")
        return None


def test_endpoint_with_gradcam(endpoint_url, api_key, image_b64):
    """Test endpoint with GradCAM."""
    print("\n" + "="*60)
    print("TEST 2: Prediction with GradCAM")
    print("="*60)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "image": image_b64,
        "include_gradcam": True
    }
    
    response = requests.post(endpoint_url, headers=headers, json=payload)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✔️ Success!")
        print(f"Predicted Label: {result.get('label')}")
        print(f"Confidence: {result.get('score', 0):.2%}")
        
        if 'gradcam' in result and result['gradcam']:
            print(f"✔️ GradCAM generated (length: {len(result['gradcam'])} chars)")
            
            # Save GradCAM image
            save_gradcam_image(result['gradcam'], "gradcam_output.png")
        else:
            print("⚠️ No GradCAM in response")
        
        return result
    else:
        print(f"❌ Error: {response.text}")
        return None


def save_gradcam_image(gradcam_b64, output_path):
    """Save GradCAM base64 string as image file."""
    try:
        image_data = base64.b64decode(gradcam_b64)
        image = Image.open(io.BytesIO(image_data))
        image.save(output_path)
        print(f"✔️ GradCAM saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving GradCAM: {e}")


def main():
    """Run all tests."""
    print("="*60)
    print("AZURE ML ENDPOINT TESTING")
    print("="*60)
    print(f"Endpoint: {ENDPOINT_URL}")
    print(f"Image: {IMAGE_PATH}")
    
    # Load image
    print("\nLoading image...")
    try:
        image_b64 = load_image_as_base64(IMAGE_PATH)
        print(f"✔️ Image loaded (base64 length: {len(image_b64)} chars)")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Test 1: Basic prediction
    result1 = test_endpoint_basic(ENDPOINT_URL, API_KEY, image_b64)
    
    # Test 2: With GradCAM
    result2 = test_endpoint_with_gradcam(ENDPOINT_URL, API_KEY, image_b64)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
