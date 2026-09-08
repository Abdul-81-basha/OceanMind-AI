import os
import tempfile
from io import BytesIO

import numpy as np
import rasterio
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OceanMind AI",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

MODEL_PATH = "marine_debris_segmentation.pth"
THRESHOLD = 0.50
EXPECTED_BANDS = 11
INPUT_SIZE = 128


# ============================================================
# U-NET MODEL
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, input_channels=11):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(input_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )
        self.dec1 = DoubleConv(64, 32)

        # Output
        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder 3
        d3 = self.up3(b)

        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(
                d3,
                size=e3.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        # Decoder 2
        d2 = self.up2(d3)

        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(
                d2,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        # Decoder 1
        d1 = self.up1(d2)

        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(
                d1,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"

    try:
        device = torch.device("cpu")

        model = UNet(input_channels=EXPECTED_BANDS)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False
        )

        # Handle different possible saving formats
        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            else:
                state_dict = checkpoint

        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        return model, None

    except Exception as e:
        return None, str(e)


# ============================================================
# NORMALIZE SATELLITE IMAGE
# ============================================================

def normalize_image(image):

    image = image.astype(np.float32)

    normalized = np.zeros_like(image, dtype=np.float32)

    for band in range(image.shape[0]):

        data = image[band]

        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        minimum = np.min(data)
        maximum = np.max(data)

        if maximum > minimum:

            normalized[band] = (
                data - minimum
            ) / (
                maximum - minimum
            )

        else:
            normalized[band] = 0.0

    return normalized


# ============================================================
# TRUE COLOR IMAGE
# Sentinel-2:
# Band 4 = Red
# Band 3 = Green
# Band 2 = Blue
#
# rasterio indexes start at 1
# ============================================================

def create_true_color(image):

    if image.shape[0] < 3:
        return None

    red = image[2].astype(np.float32)
    green = image[1].astype(np.float32)
    blue = image[0].astype(np.float32)

    rgb = np.stack(
        [red, green, blue],
        axis=-1
    )

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Percentile stretching
    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    if high > low:

        rgb = (
            rgb - low
        ) / (
            high - low
        )

    rgb = np.clip(rgb, 0, 1)

    rgb = (rgb * 255).astype(np.uint8)

    return rgb


# ============================================================
# CREATE MASK IMAGE
# ============================================================

def create_mask_image(mask):

    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )

    return Image.fromarray(
        mask_uint8,
        mode="L"
    )


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(rgb, mask):

    base = Image.fromarray(rgb).convert("RGBA")

    overlay = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            4
        ),
        dtype=np.uint8
    )

    # Red detection region
    overlay[mask == 1] = [
        255,
        0,
        0,
        150
    ]

    overlay_image = Image.fromarray(
        overlay,
        mode="RGBA"
    )

    result = Image.alpha_composite(
        base,
        overlay_image
    )

    return result.convert("RGB")


# ============================================================
# SEVERITY
# ============================================================

def get_severity(area_percentage):

    if area_percentage == 0:
        return "No Detection"

    elif area_percentage < 0.10:
        return "Low"

    elif area_percentage < 1.0:
        return "Moderate"

    else:
        return "High"


# ============================================================
# HEADER
# ============================================================

st.title("🌊 OceanMind AI")

st.subheader(
    "AI-Based Marine Pollution Detection & Monitoring"
)

st.write(
    "Detect potential marine debris using "
    "11-band Sentinel-2 satellite imagery "
    "and a U-Net deep learning model."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌊 OceanMind AI")

    st.write(
        "Marine pollution detection "
        "using satellite imagery."
    )

    st.divider()

    st.subheader("System Information")

    st.write("**Model:** U-Net CNN")
    st.write("**Input Bands:** 11")
    st.write("**AI Input Size:** 128 × 128")
    st.write("**Output:** Debris Segmentation Mask")
    st.write("**Threshold:** 0.50")
    st.write("**Device:** CPU")

    st.divider()

    st.subheader("Project Status")

    st.success("Model Ready")
    st.success("Dashboard Ready")

    st.divider()

    st.caption(
        "Research prototype for marine "
        "pollution monitoring."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📡 Upload Satellite Image")

uploaded_file = st.file_uploader(
    "Upload an 11-band Sentinel-2 GeoTIFF (.tif / .tiff)",
    type=["tif", "tiff"]
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    temp_path = None

    try:

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".tif"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        # Read GeoTIFF
        with rasterio.open(temp_path) as src:

            image = src.read()

            crs = src.crs

            bounds = src.bounds

            width = src.width

            height = src.height

            band_count = src.count

        # ----------------------------------------------------
        # IMAGE INFORMATION
        # ----------------------------------------------------

        st.header("🛰️ Image Information")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Spectral Bands",
            band_count
        )

        col2.metric(
            "Width",
            width
        )

        col3.metric(
            "Height",
            height
        )

        col4.metric(
            "AI Input",
            "128 × 128"
        )

        # ----------------------------------------------------
        # BAND VALIDATION
        # ----------------------------------------------------

        if band_count != EXPECTED_BANDS:

            st.error(
                f"Expected {EXPECTED_BANDS} bands, "
                f"but received {band_count} bands."
            )

            st.stop()

        st.success(
            "Valid 11-band Sentinel-2 image detected."
        )

        # ----------------------------------------------------
        # GEOSPATIAL INFORMATION
        # ----------------------------------------------------

        with st.expander("🌍 Geospatial Information"):

            st.write(
                f"**CRS:** {crs}"
            )

            st.write(
                f"**Bounds:** {bounds}"
            )

        # ----------------------------------------------------
        # TRUE COLOR
        # ----------------------------------------------------

        rgb = create_true_color(image)

        st.header("🖼️ Satellite Image")

        st.image(
            rgb,
            caption="Sentinel-2 True Color Composite",
            use_container_width=True
        )

        # ----------------------------------------------------
        # ANALYZE BUTTON
        # ----------------------------------------------------

        st.divider()

        analyze = st.button(
            "🔍 Analyze Marine Pollution",
            type="primary",
            use_container_width=True
        )

        if analyze:

            model, error = load_model()

            if error is not None:

                st.error(
                    f"Unable to load model: {error}"
                )

                st.stop()

            # ------------------------------------------------
            # PREPROCESS
            # ------------------------------------------------

            normalized = normalize_image(image)

            tensor = torch.from_numpy(
                normalized
            ).float()

            tensor = tensor.unsqueeze(0)

            # Resize to model input size
            tensor = F.interpolate(
                tensor,
                size=(INPUT_SIZE, INPUT_SIZE),
                mode="bilinear",
                align_corners=False
            )

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            with st.spinner(
                "AI model is analyzing the image..."
            ):

                with torch.no_grad():

                    output = model(tensor)

                    probability = torch.sigmoid(
                        output
                    )

            probability = probability.squeeze().cpu().numpy()

            # ------------------------------------------------
            # RESIZE PREDICTION TO ORIGINAL IMAGE SIZE
            # ------------------------------------------------

            probability_original = np.array(
                Image.fromarray(
                    probability.astype(np.float32),
                    mode="F"
                ).resize(
                    (width, height),
                    Image.Resampling.BILINEAR
                )
            )

            # ------------------------------------------------
            # CREATE BINARY MASK
            # ------------------------------------------------

            mask = (
                probability_original >= THRESHOLD
            ).astype(np.uint8)

            # ------------------------------------------------
            # CALCULATE RESULTS
            # ------------------------------------------------

            debris_pixels = int(
                np.sum(mask)
            )

            total_pixels = mask.size

            area_percentage = (
                debris_pixels /
                total_pixels
            ) * 100

            severity = get_severity(
                area_percentage
            )

            # ------------------------------------------------
            # AI CONFIDENCE ESTIMATE
            # ------------------------------------------------

            detected_probabilities = (
                probability_original[mask == 1]
            )

            if len(detected_probabilities) > 0:

                confidence = (
                    float(
                        np.mean(
                            detected_probabilities
                        )
                    ) * 100
                )

            else:

                confidence = (
                    float(
                        np.max(
                            probability_original
                        )
                    ) * 100
                )

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            st.header("🚨 Detection Result")

            if debris_pixels > 0:

                st.warning(
                    "Potential marine debris detected."
                )

            else:

                st.success(
                    "No potential marine debris detected."
                )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Detected Area",
                f"{area_percentage:.2f}%"
            )

            col2.metric(
                "Debris Pixels",
                f"{debris_pixels:,}"
            )

            col3.metric(
                "AI Confidence",
                f"{confidence:.1f}%"
            )

            col4.metric(
                "Severity",
                severity
            )

            # ------------------------------------------------
            # WARNING ABOUT CONFIDENCE
            # ------------------------------------------------

            st.info(
                "AI Confidence is the average predicted "
                "probability of detected pixels. It is a "
                "model-confidence estimate, not validated "
                "detection accuracy."
            )

            # ------------------------------------------------
            # VISUAL RESULTS
            # ------------------------------------------------

            st.header("📊 Segmentation Results")

            mask_image = create_mask_image(
                mask
            )

            overlay_image = create_overlay(
                rgb,
                mask
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    mask_image,
                    caption="Marine Debris Segmentation Mask",
                    use_container_width=True
                )

            with col2:

                st.image(
                    overlay_image,
                    caption="Detection Overlay",
                    use_container_width=True
                )

            # ------------------------------------------------
            # DOWNLOAD MASK
            # ------------------------------------------------

            mask_buffer = BytesIO()

            mask_image.save(
                mask_buffer,
                format="PNG"
            )

            st.download_button(
                label="⬇️ Download Segmentation Mask",
                data=mask_buffer.getvalue(),
                file_name="predicted_debris_mask.png",
                mime="image/png",
                use_container_width=True
            )

            # ------------------------------------------------
            # DOWNLOAD OVERLAY
            # ------------------------------------------------

            overlay_buffer = BytesIO()

            overlay_image.save(
                overlay_buffer,
                format="PNG"
            )

            st.download_button(
                label="⬇️ Download Detection Overlay",
                data=overlay_buffer.getvalue(),
                file_name="debris_overlay.png",
                mime="image/png",
                use_container_width=True
            )

            # ------------------------------------------------
            # REPORT
            # ------------------------------------------------

            report = f"""
OceanMind AI - Marine Pollution Detection Report
=================================================

Input Image
-----------
Filename: {uploaded_file.name}
Bands: {band_count}
Image Size: {width} x {height}

AI Model
--------
Model: U-Net CNN
Input Channels: {EXPECTED_BANDS}
AI Input Size: {INPUT_SIZE} x {INPUT_SIZE}
Detection Threshold: {THRESHOLD}

Detection Results
-----------------
Debris Pixels: {debris_pixels:,}
Total Pixels: {total_pixels:,}
Detected Area: {area_percentage:.2f}%
AI Confidence Estimate: {confidence:.1f}%
Severity: {severity}

Important Note
--------------
The AI confidence value represents the model's
predicted probability for detected pixels.
It is not a validated real-world accuracy measure.

OceanMind AI is a research prototype.
"""

            st.download_button(
                label="📄 Download Analysis Report",
                data=report,
                file_name="oceanmind_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )

            # ------------------------------------------------
            # PIPELINE
            # ------------------------------------------------

            st.divider()

            st.header("⚙️ AI Analysis Pipeline")

            st.write(
                """
                **1. Satellite Image**
                → 11-band Sentinel-2 GeoTIFF

                **2. Preprocessing**
                → Band-wise normalization

                **3. Resizing**
                → 128 × 128 AI input

                **4. U-Net CNN**
                → Encoder + Bottleneck + Decoder

                **5. Pixel-wise Prediction**
                → Marine debris probability map

                **6. Thresholding**
                → Probability ≥ 0.50

                **7. Output**
                → Marine debris segmentation mask
                """
            )

    except Exception as e:

        st.error(
            f"Error processing the uploaded image: {e}"
        )

    finally:

        if temp_path is not None:

            try:
                os.remove(temp_path)
            except:
                pass


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "OceanMind AI | AI-Based Marine Pollution Detection "
    "using Sentinel-2 Satellite Imagery"
)

st.caption(
    "Research prototype — results should be "
    "validated before real-world environmental decisions."
)
