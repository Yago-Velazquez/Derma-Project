import os
import json
import base64
import io

from PIL import Image
import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Azure ML sets this environment variable with the base model path
MODEL_DIR = os.getenv("AZUREML_MODEL_DIR")

# Global objects loaded in init()
processor = None
model = None
id2label = None


def init():
    """
    Executed once when the container starts.
    Loads the model and processor.
    """
    global processor, model, id2label

    base_model_dir = MODEL_DIR
    
    print(f"AZUREML_MODEL_DIR: {base_model_dir}")
    
    # List contents to debug
    if os.path.exists(base_model_dir):
        print(f"Contents of {base_model_dir}:")
        for root, dirs, files in os.walk(base_model_dir):
            level = root.replace(base_model_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f'{subindent}{file}')
    
    # Check for subdirectory (skin_cancer_model or model_2_clean)
    subdirs = ["skin_cancer_model", "model_2_clean"]
    model_path = base_model_dir
    
    for subdir in subdirs:
        potential_path = os.path.join(base_model_dir, subdir)
        if os.path.isdir(potential_path):
            model_path = potential_path
            print(f"Found model in subdirectory: {subdir}")
            break
    
    print(f"Loading model from: {model_path}")
    
    # Check if required files exist
    config_path = os.path.join(model_path, "config.json")
    preprocessor_path = os.path.join(model_path, "preprocessor_config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"preprocessor_config.json not found at {preprocessor_path}")
    
    print(f"Found config.json: {os.path.exists(config_path)}")
    print(f"Found preprocessor_config.json: {os.path.exists(preprocessor_path)}")

    # Load processor and model with attention outputs for GradCAM
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    # Try PyTorch format first (more stable)
    try:
        model = AutoModelForImageClassification.from_pretrained(
            model_path,
            use_safetensors=False,
            output_attentions=True
        )
        print("Model loaded using PyTorch format")
    except Exception as e:
        print(f"PyTorch load failed, trying safetensors: {e}")
        model = AutoModelForImageClassification.from_pretrained(
            model_path,
            output_attentions=True
        )
        print("Model loaded using SafeTensors format")
    
    # Set model to evaluation mode and eager attention
    model.eval()
    try:
        model.set_attn_implementation("eager")
    except:
        print("Could not set attention implementation to eager")

    # Get id to label mapping
    id2label = model.config.id2label if hasattr(model.config, "id2label") else None

    print("Model and processor loaded successfully.")
    if id2label:
        print(f"Classes: {id2label}")


def _load_image_from_base64(b64_str: str) -> Image.Image:
    """
    Decodes a base64 image string to a PIL.Image object.
    """
    image_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img


def overlay_heatmap(heatmap, image, alpha=0.6):
    """
    Overlay heatmap on top of original image.
    """
    heatmap = cv2.resize(heatmap, image.size)
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)

    image_np = np.array(image.convert("RGB"))[..., ::-1]  # RGB->BGR
    overlay = cv2.addWeighted(heatmap, alpha, image_np, 1 - alpha, 0)
    overlay = overlay[..., ::-1]  # BGR->RGB

    overlay_img = Image.fromarray(overlay)

    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def compute_cam_multi(attentions, last_k=4):
    """
    Compute multi-layer GradCAM from attentions.
    """
    cams = []
    for attn in attentions[-last_k:]:
        grad = attn.grad
        if grad is None:
            continue
        attn_cls = attn[:, :, 0, 1:]
        grad_cls = grad[:, :, 0, 1:]
        cam = (attn_cls * grad_cls).mean(dim=1)
        cam = cam[0].detach().cpu().numpy()
        cams.append(cam)

    if not cams:
        return None

    cam = np.mean(cams, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    side = int(cam.size ** 0.5)
    return cam.reshape(side, side)


def generate_gradcam(img: Image.Image, cls_id: int):
    """
    Generate GradCAM heatmap for the predicted class.
    """
    try:
        # Prepare inputs with gradients
        inputs = processor(images=img, return_tensors="pt")
        inputs["pixel_values"].requires_grad_()

        # Forward pass with attentions
        out = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )

        attentions = out.attentions
        last_k = min(4, len(attentions))

        # Retain gradients for attention layers
        for attn in attentions[-last_k:]:
            attn.retain_grad()

        # Backward pass on predicted class
        logit = out.logits[0]
        logit[cls_id].backward()

        # Compute CAM
        cam = compute_cam_multi(attentions, last_k)
        
        if cam is None:
            return None
        
        # Overlay on original image
        gradcam_b64 = overlay_heatmap(cam, img)
        return gradcam_b64

    except Exception as e:
        print(f"Error generating GradCAM: {e}")
        import traceback
        traceback.print_exc()
        return None


def run(raw_data):
    """
    Endpoint function for inference.
    
    Expected input JSON:
        {
            "image": "<base64_encoded_image>",
            "include_gradcam": true/false (optional, default: false)
        }
    
    Returns:
        {
            "label": "predicted_class_name",
            "score": confidence_score,
            "all_scores": [{"label": "class1", "score": 0.xx}, ...],
            "gradcam": "<base64_encoded_heatmap_image>" (if requested)
        }
    """
    try:
        # Parse input
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        if "image" not in data:
            return {"error": "Missing 'image' field in input JSON."}

        b64_image = data["image"]
        include_gradcam = data.get("include_gradcam", False)

        # Load image from base64
        img = _load_image_from_base64(b64_image)

        # Preprocess image
        inputs = processor(images=img, return_tensors="pt")

        # Run inference (without gradients for prediction)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        # Get top prediction
        top_prob, top_idx = torch.max(probs, dim=-1)
        top_idx = int(top_idx)

        if id2label is not None:
            top_label = id2label.get(top_idx, str(top_idx))
        else:
            top_label = str(top_idx)

        # Get all class probabilities
        all_scores = []
        num_classes = probs.shape[0]
        for i in range(num_classes):
            label_i = id2label.get(i, str(i)) if id2label is not None else str(i)
            all_scores.append({
                "label": label_i,
                "score": float(probs[i])
            })
        
        # Sort by score descending
        all_scores.sort(key=lambda x: x["score"], reverse=True)

        result = {
            "label": top_label,
            "score": float(top_prob),
            "all_scores": all_scores
        }

        # Generate GradCAM if requested
        if include_gradcam:
            gradcam_b64 = generate_gradcam(img, top_idx)
            result["gradcam"] = gradcam_b64

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }