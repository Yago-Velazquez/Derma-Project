"""
Simulate Azure Function Call
Sends an image to your Azure Function and displays the full response
"""

import requests
import json
import base64
from PIL import Image
import io
from datetime import datetime

# Configuration
FUNCTION_URL = "https://func-app-derma-dxhpckc7c8gfake3.spaincentral-01.azurewebsites.net/api/InferenciaFunction?code=o0_9STSWGjwmEt4LJwr17gwEadQzI3uEuflmFw6_r4tuAzFufl8__w=="
IMAGE_PATH = "/Users/yago.velazquez/Derma-Project/classifiers/modelo-2/sample.jpg"


def load_image():
    """Load test image."""
    with open(IMAGE_PATH, 'rb') as f:
        return f.read()


def save_gradcam(gradcam_b64, output_path="gradcam_output.png"):
    """Save GradCAM base64 image to file."""
    try:
        image_data = base64.b64decode(gradcam_b64)
        image = Image.open(io.BytesIO(image_data))
        image.save(output_path)
        print(f"  ✔️ GradCAM saved to: {output_path}")
        return True
    except Exception as e:
        print(f"  ❌ Error saving GradCAM: {e}")
        return False


def simulate_function_call():
    """Simulate a call to the Azure Function."""
    
    print("="*70)
    print("SIMULATING AZURE FUNCTION CALL")
    print("="*70)
    print(f"Function URL: {FUNCTION_URL}")
    print(f"Image: {IMAGE_PATH}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load image
    print("\n[1/3] Loading image...")
    try:
        image_data = load_image()
        img = Image.open(IMAGE_PATH)
        print(f"  ✔️ Image loaded: {len(image_data)} bytes")
        print(f"  ✔️ Dimensions: {img.size[0]}x{img.size[1]} pixels")
        print(f"  ✔️ Format: {img.format}")
        print(f"  ✔️ Mode: {img.mode}")
    except Exception as e:
        print(f"  ❌ Error loading image: {e}")
        return None
    
    # Send request
    print("\n[2/3] Sending request to Azure Function...")
    print(f"  → POST {FUNCTION_URL[:80]}...")
    print(f"  → Sending {len(image_data)} bytes...")
    
    try:
        start_time = datetime.now()
        
        response = requests.post(
            FUNCTION_URL,
            headers={"Content-Type": "application/octet-stream"},
            data=image_data,
            timeout=120
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"  ✔️ Response received in {duration:.2f} seconds")
        print(f"  ✔️ Status code: {response.status_code}")
        
    except requests.exceptions.Timeout:
        print("  ❌ Request timed out (>120 seconds)")
        print("\n  Possible causes:")
        print("    - External APIs (Azure ML or X.AI) are slow")
        print("    - Cold start taking too long")
        print("    - Network issues")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"  ❌ Connection error: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None
    
    # Process response
    print("\n[3/3] Processing response...")
    
    if response.status_code == 200:
        try:
            result = response.json()
            print("  ✔️ Response parsed successfully\n")
            
            # Display results
            print("="*70)
            print("ANALYSIS RESULTS")
            print("="*70)
            
            # Lesion Detection
            is_lesion = result.get('is_lesion', False)
            probability = result.get('probability', 0)
            
            print(f"\n🔍 LESION DETECTION:")
            print(f"   Result: {'✔️ YES - Lesion detected' if is_lesion else '❌ NO - No lesion detected'}")
            print(f"   Confidence: {probability:.2%}")
            
            # Progress bar for confidence
            bar_length = 30
            filled = int(bar_length * probability)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"   [{bar}] {probability:.1%}")
            
            # Lesion Classification
            lesion_type = result.get('lesion_type')
            lesion_prob = result.get('lesion_type_probability', 0)
            
            print(f"\n🏥 LESION CLASSIFICATION:")
            if lesion_type:
                print(f"   Type: {lesion_type}")
                print(f"   Confidence: {lesion_prob:.2%}")
                
                # Progress bar for lesion type confidence
                filled = int(bar_length * lesion_prob)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"   [{bar}] {lesion_prob:.1%}")
            else:
                print(f"   Status: Not performed")
                print(f"   Reason: Probability below threshold or no lesion detected")
            
            # GradCAM
            gradcam = result.get('gradcam')
            print(f"\n🎨 GRADCAM HEATMAP:")
            if gradcam:
                print(f"   Generated: ✔️ YES")
                print(f"   Data size: {len(gradcam):,} characters")
                
                # Save GradCAM
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"gradcam_{timestamp}.png"
                if save_gradcam(gradcam, output_path):
                    print(f"   Preview: Open {output_path} to view heatmap")
            else:
                print(f"   Generated: ❌ NO")
                print(f"   Note: GradCAM only generated when lesion is detected")
            
            # AI Explanation
            explanation = result.get('explanation', '')
            if explanation:
                print(f"\n💬 AI EXPLANATION:")
                print("-" * 70)
                # Word wrap at 70 characters
                words = explanation.split()
                line = "   "
                for word in words:
                    if len(line) + len(word) + 1 > 70:
                        print(line)
                        line = "   " + word
                    else:
                        line += (" " + word) if line != "   " else word
                if line.strip():
                    print(line)
                print("-" * 70)
            
            # Raw JSON
            print("\n" + "="*70)
            print("RAW JSON RESPONSE")
            print("="*70)
            # Pretty print JSON without gradcam (too long)
            display_result = result.copy()
            if 'gradcam' in display_result and display_result['gradcam']:
                display_result['gradcam'] = f"<base64 data: {len(display_result['gradcam'])} chars>"
            print(json.dumps(display_result, indent=2, ensure_ascii=False))
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"  ❌ Error: Invalid JSON response")
            print(f"     {e}")
            print(f"\n  Raw response (first 500 chars):")
            print(f"  {response.text[:500]}")
            return None
            
    elif response.status_code == 500:
        print(f"  ❌ 500 Internal Server Error")
        print(f"\n  Response body:")
        print(f"  {response.text}")
        
        print("\n" + "="*70)
        print("TROUBLESHOOTING")
        print("="*70)
        print("""
Most common causes:

1. MISSING ENVIRONMENT VARIABLES:
   Check Azure Portal → Function App → Configuration → Application settings
   
   Required:
   - MODEL2_URL: Your Azure ML endpoint URL
   - MODEL2_TOKEN: Your Azure ML authentication key
   - XAI_API_KEY: Your X.AI/Grok API key
   
   Optional:
   - XAI_BASE_URL: https://api.x.ai/v1
   - XAI_MODEL: grok-vision-beta
   - THRESHOLD: 0.7

2. AZURE ML ENDPOINT ISSUES:
   - Endpoint not running
   - MODEL2_URL is incorrect
   - MODEL2_TOKEN is invalid/expired

3. X.AI API ISSUES:
   - XAI_API_KEY is invalid/expired
   - API quota exceeded
   - Network connectivity issues

HOW TO FIX:
1. Set environment variables in Azure Portal
2. Click 'Save' and restart the Function App
3. Check Function logs: Portal → Function App → Log stream
4. Test Azure ML endpoint independently
5. Verify X.AI API key is valid
        """)
        return None
        
    elif response.status_code == 401:
        print(f"  ❌ 401 Unauthorized")
        print("\n  The function key in the URL is incorrect or expired")
        print("  Get the correct key from: Azure Portal → Function App → Functions → inferencia → Get Function URL")
        return None
        
    elif response.status_code == 404:
        print(f"  ❌ 404 Not Found")
        print("\n  The function URL is incorrect")
        print("  Check: Function name should be 'inferencia'")
        return None
        
    else:
        print(f"  ❌ Unexpected status code: {response.status_code}")
        print(f"\n  Response body:")
        print(f"  {response.text[:500]}")
        return None


def main():
    """Main function."""
    
    result = simulate_function_call()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    if result:
        print("✔️ Function call successful!")
        print(f"✔️ Lesion detected: {result.get('is_lesion', False)}")
        if result.get('lesion_type'):
            print(f"✔️ Lesion type: {result.get('lesion_type')}")
        if result.get('gradcam'):
            print(f"✔️ GradCAM heatmap generated")
        print(f"✔️ AI explanation provided")
    else:
        print("❌ Function call failed")
        print("\nNext steps:")
        print("1. Check the error message above")
        print("2. View Function logs in Azure Portal")
        print("3. Verify environment variables are set")
        print("4. Test Azure ML endpoint independently")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
