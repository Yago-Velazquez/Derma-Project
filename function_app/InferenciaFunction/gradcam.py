from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import torch
import base64
from io import BytesIO
import cv2

# ============================================
# 🔥 Load HuggingFace model ONCE (cold start)
# ============================================
model_id = "Anwarkh1/Skin_Cancer-Image_Classification"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(
    model_id, output_attentions=True
)
model.set_attn_implementation("eager")
model.eval()


# ============================================
# 🔧 Overlay heatmap on top of original image
# ============================================
def overlay_heatmap(heatmap, image, alpha=0.6):
    heatmap = cv2.resize(heatmap, image.size)
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)

    image_np = np.array(image.convert("RGB"))[..., ::-1]  # RGB->BGR
    overlay = cv2.addWeighted(heatmap, alpha, image_np, 1 - alpha, 0)
    overlay = overlay[..., ::-1]  # BGR->RGB

    overlay_img = Image.fromarray(overlay)

    buf = BytesIO()
    overlay_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ============================================
# 🔧 Compute multi-layer GradCAM from attentions
# ============================================
def compute_cam_multi(attentions, last_k=4):
    cams = []
    for attn in attentions[-last_k:]:
        grad = attn.grad
        attn_cls = attn[:, :, 0, 1:]
        grad_cls = grad[:, :, 0, 1:]
        cam = (attn_cls * grad_cls).mean(dim=1)
        cam = cam[0].detach().cpu().numpy()
        cams.append(cam)

    cam = np.mean(cams, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    side = int(cam.size ** 0.5)
    return cam.reshape(side, side)


# ============================================
# 🔥 Main GradCAM function for Azure Function
# ============================================
def analyze_image_bytes(image_bytes: bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # First forward pass (no gradients)
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out.logits, dim=1)
        cls_id = torch.argmax(probs, dim=1).item()
        prob = probs[0, cls_id].item()

    # Second forward pass with gradients
    inputs = processor(images=img, return_tensors="pt")
    inputs["pixel_values"].requires_grad_()

    out = model(
        **inputs,
        output_attentions=True,
        return_dict=True
    )

    attentions = out.attentions
    last_k = min(4, len(attentions))

    for attn in attentions[-last_k:]:
        attn.retain_grad()

    logit = out.logits[0]
    logit[cls_id].backward()

    cam = compute_cam_multi(attentions, last_k)
    b64 = overlay_heatmap(cam, img)

    return cls_id, prob, b64