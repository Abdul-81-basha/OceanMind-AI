from pathlib import Path
import sys
import uuid
import io

import numpy as np
import rasterio
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "backend" / "outputs"
MODEL_PATH = BASE_DIR / "outputs" / "marine_debris_segmentation.pth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from unet_model import UNet

app = FastAPI(title="OceanMind AI 2.0")

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# Model IDs 0-14 correspond to MARIDA IDs 1-15.
CLASS_NAMES = [
    "Marine Debris",
    "Dense Sargassum",
    "Sparse Sargassum",
    "Natural Organic Material",
    "Ship",
    "Clouds",
    "Marine Water",
    "Sediment-Laden Water",
    "Foam",
    "Turbid Water",
    "Shallow Water",
    "Waves",
    "Cloud Shadows",
    "Wakes",
    "Mixed Water",
]

# Colors indexed by MARIDA ID. ID 1 (debris) is red.
CLASS_COLORS = {
    1: (230, 45, 45),
    2: (230, 130, 30),
    3: (240, 200, 40),
    4: (150, 90, 45),
    5: (220, 50, 180),
    6: (170, 190, 220),
    7: (30, 100, 220),
    8: (50, 190, 200),
    9: (240, 240, 240),
    10: (170, 100, 210),
    11: (50, 170, 90),
    12: (40, 200, 160),
    13: (100, 100, 100),
    14: (230, 100, 140),
    15: (100, 140, 40),
}


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        print(f"Model checkpoint not found: {MODEL_PATH}")
        return

    net = UNet(in_channels=11, num_classes=15).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    net.load_state_dict(state_dict)
    net.eval()
    model = net
    print(f"OceanMind U-Net loaded on {DEVICE}")


@app.get("/")
def home():
    return {"message": "OceanMind AI 2.0 backend is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


def make_rgb_preview(image):
    # Sentinel-2-style RGB: bands 4, 3, 2 (1-based).
    # This assumes the uploaded GeoTIFF follows that band ordering.
    rgb = image[[3, 2, 1]].astype(np.float32)
    channels = []

    for band in rgb:
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            channels.append(np.zeros(band.shape, dtype=np.uint8))
            continue

        low, high = np.percentile(valid, [2, 98])
        scaled = (band - low) / max(float(high - low), 1e-6)
        channels.append((np.clip(scaled, 0, 1) * 255).astype(np.uint8))

    return np.stack(channels, axis=-1)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sensor_type: str = Form("optical"),
):
    if sensor_type != "optical":
        raise HTTPException(400, "This checkpoint supports 11-band optical input only.")

    if model is None:
        raise HTTPException(503, "Model checkpoint could not be loaded.")

    filename = (file.filename or "").lower()
    if not filename.endswith((".tif", ".tiff")):
        raise HTTPException(
            400,
            "Upload an 11-band MARIDA-style GeoTIFF (.tif or .tiff). "
            "JPEG/PNG files are not supported model inputs.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty.")

    image_id = uuid.uuid4().hex[:10]

    try:
        with rasterio.MemoryFile(content) as memfile:
            with memfile.open() as src:
                image = src.read().astype(np.float32)

        if image.shape != (11, 256, 256):
            raise HTTPException(400, f"Expected 11 bands at 256x256; got {image.shape}.")

        if not np.isfinite(image).all():
            raise HTTPException(400, "Image contains invalid numeric values.")

        # Save the original-image visualization before model normalization.
        rgb = make_rgb_preview(image)
        original_path = OUTPUT_DIR / f"{image_id}_original.png"
        Image.fromarray(rgb, mode="RGB").save(original_path)

        # Match the training loader's per-band min-max normalization.
        lo = image.min(axis=(1, 2), keepdims=True)
        hi = image.max(axis=(1, 2), keepdims=True)
        normalized = (image - lo) / np.maximum(hi - lo, 1e-6)

        tensor = torch.from_numpy(normalized.copy()).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = logits.argmax(dim=1)[0].cpu().numpy()
            confidence = probabilities.max(dim=1)[0].mean().item()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Could not read or process the GeoTIFF: {exc}")

    # Convert internal IDs 0-14 to MARIDA IDs 1-15 for visualization.
    marida_mask = (prediction + 1).astype(np.uint8)

    mask_path = OUTPUT_DIR / f"{image_id}_mask.png"
    Image.fromarray(marida_mask, mode="L").save(mask_path)

    # Color mask and blend it with the RGB preview.
    color_mask = np.zeros_like(rgb)
    for class_id, color in CLASS_COLORS.items():
        color_mask[marida_mask == class_id] = color

    overlay = (
        rgb.astype(np.float32) * 0.55
        + color_mask.astype(np.float32) * 0.45
    ).clip(0, 255).astype(np.uint8)

    overlay_path = OUTPUT_DIR / f"{image_id}_overlay.png"
    Image.fromarray(overlay, mode="RGB").save(overlay_path)

    unique, counts = np.unique(marida_mask, return_counts=True)
    total = marida_mask.size
    percentages = {
        CLASS_NAMES[int(class_id) - 1]: round(int(count) / total * 100, 2)
        for class_id, count in zip(unique, counts)
    }

    debris_percent = percentages.get("Marine Debris", 0.0)

    return {
        "message": (
            "Experimental model prediction only. Results require independent "
            "validation before environmental use."
        ),
        "pollution_type": (
            "Marine Debris predicted"
            if debris_percent > 0
            else "Marine Debris not predicted"
        ),
        "detected_class": CLASS_NAMES[int(np.bincount(prediction.ravel()).argmax())],
        "severity": (
            "High" if debris_percent >= 10
            else "Moderate" if debris_percent >= 3
            else "Low"
        ),
        "pollution_area_percent": debris_percent,
        "confidence": round(confidence * 100, 2),
        "processed_image_url": f"/outputs/{image_id}_original.png",
        "mask_url": f"/outputs/{image_id}_mask.png",
        "output_image_url": f"/outputs/{image_id}_overlay.png",
        "class_percentages": percentages,
        "sensor_type": sensor_type,
    }


