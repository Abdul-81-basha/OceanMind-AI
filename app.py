import os
import tempfile
from io import BytesIO

import numpy as np
import rasterio
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OceanMind AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "outputs/marine_debris_segmentation.pth"

IMAGE_SIZE = 128

THRESHOLD = 0.50


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                output_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                output_channels,
                output_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                output_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )


    def forward(self, x):

        return self.block(x)


class UNet(nn.Module):

    def __init__(
        self,
        input_channels=11
    ):

        super().__init__()

        self.enc1 = DoubleConv(
            input_channels,
            32
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            128,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.final = nn.Conv2d(
            32,
            1,
            1
        )


    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = UNet(
        input_channels=11
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):

    image = image.astype(
        np.float32
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    for band in range(
        image.shape[0]
    ):

        data = image[band]

        minimum = np.min(data)

        maximum = np.max(data)

        if (
            np.isfinite(minimum)
            and
            np.isfinite(maximum)
            and
            maximum > minimum
        ):

            image[band] = (
                data - minimum
            ) / (
                maximum - minimum
            )

        else:

            image[band] = 0.0

    return np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )


# ============================================================
# RGB CREATION
# ============================================================

def create_rgb(image):

    rgb = image[
        [3, 2, 1]
    ]

    rgb = np.transpose(
        rgb,
        (1, 2, 0)
    )

    for channel in range(3):

        data = rgb[:, :, channel]

        minimum = np.percentile(
            data,
            2
        )

        maximum = np.percentile(
            data,
            98
        )

        if maximum > minimum:

            rgb[:, :, channel] = (
                data - minimum
            ) / (
                maximum - minimum
            )

        else:

            rgb[:, :, channel] = 0.0

    return np.clip(
        rgb,
        0.0,
        1.0
    )


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    rgb,
    mask
):

    overlay = rgb.copy()

    overlay[mask] = [
        1.0,
        0.0,
        0.0
    ]

    return overlay


# ============================================================
# PNG CONVERTER
# ============================================================

def array_to_png_bytes(
    array,
    is_mask=False
):

    if is_mask:

        image = Image.fromarray(
            (
                array.astype(
                    np.uint8
                ) * 255
            )
        )

    else:

        image = Image.fromarray(
            (
                np.clip(
                    array,
                    0,
                    1
                ) * 255
            ).astype(
                np.uint8
            )
        )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# ============================================================
# SEVERITY
# ============================================================

def get_severity(
    percentage
):

    if percentage == 0:

        return (
            "No Detection",
            "No potential debris pixels detected."
        )

    elif percentage < 0.10:

        return (
            "Low",
            "Small potential debris region detected."
        )

    elif percentage < 1.0:

        return (
            "Moderate",
            "Potential debris detected over a moderate area."
        )

    else:

        return (
            "High",
            "Large potential debris region detected."
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌊 OceanMind AI"
)

st.subheader(
    "AI-Based Marine Pollution Detection & Monitoring"
)

st.write(
    "An AI-powered satellite image analysis system "
    "for identifying potential marine debris."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🌊 OceanMind AI"
    )

    st.caption(
        "Marine Pollution Intelligence Platform"
    )

    st.divider()

    st.subheader(
        "🧠 AI System"
    )

    st.write(
        "**Model:** U-Net CNN"
    )

    st.write(
        "**Input:** 11-band Sentinel-2"
    )

    st.write(
        "**Processing:** 128 × 128"
    )

    st.write(
        "**Threshold:** 0.50"
    )

    st.write(
        f"**Device:** {device}"
    )

    st.divider()

    st.subheader(
        "📋 Development Status"
    )

    st.write(
        "✅ Dataset Preparation"
    )

    st.write(
        "✅ Model Training"
    )

    st.write(
        "✅ Model Evaluation"
    )

    st.write(
        "✅ Threshold Analysis"
    )

    st.write(
        "✅ Dashboard V1"
    )

    st.write(
        "🔄 Dashboard V2"
    )

    st.divider()

    st.caption(
        "OceanMind AI Research Prototype"
    )


# ============================================================
# MODEL CHECK
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    st.error(
        "Trained model not found."
    )

    st.write(
        f"Expected file: `{MODEL_PATH}`"
    )

    st.stop()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.header(
    "🛰️ Satellite Image Analysis"
)

st.write(
    "Upload an 11-band Sentinel-2 GeoTIFF image."
)

uploaded_file = st.file_uploader(
    "Choose satellite image",
    type=[
        "tif",
        "tiff"
    ]
)


# ============================================================
# WAIT FOR UPLOAD
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a satellite image to begin."
    )

    st.markdown(
        """
        ### 🔄 OceanMind AI Pipeline

        **Satellite Data**
        ↓

        **11-Band Preprocessing**
        ↓

        **U-Net Deep Learning**
        ↓

        **Pixel-Level Segmentation**
        ↓

        **Marine Debris Detection**
        ↓

        **Pollution Assessment**
        """
    )

    st.stop()


# ============================================================
# TEMPORARY FILE
# ============================================================

suffix = os.path.splitext(
    uploaded_file.name
)[1]


temporary = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=suffix
)

temporary.write(
    uploaded_file.getbuffer()
)

temporary.close()


# ============================================================
# READ GEOTIFF
# ============================================================

try:

    with rasterio.open(
        temporary.name
    ) as src:

        image = src.read()

        width = src.width

        height = src.height

        bands = src.count

        crs = src.crs

        bounds = src.bounds

        transform = src.transform

except Exception as error:

    st.error(
        "Could not read the GeoTIFF."
    )

    st.exception(error)

    os.unlink(
        temporary.name
    )

    st.stop()


try:

    os.unlink(
        temporary.name
    )

except:

    pass


# ============================================================
# IMAGE INFORMATION
# ============================================================

st.header(
    "📊 Satellite Data Information"
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Bands",
        bands
    )

with c2:

    st.metric(
        "Width",
        width
    )

with c3:

    st.metric(
        "Height",
        height
    )

with c4:

    st.metric(
        "AI Input",
        "128 × 128"
    )


# ============================================================
# VALIDATION
# ============================================================

if bands != 11:

    st.error(
        f"❌ This model requires 11 bands. "
        f"Uploaded image has {bands} bands."
    )

    st.stop()


st.success(
    "✅ Valid 11-band satellite image."
)


# ============================================================
# GEO INFORMATION
# ============================================================

with st.expander(
    "📍 Geospatial Information"
):

    st.write(
        f"**Coordinate Reference System:** "
        f"{crs}"
    )

    st.write(
        f"**Bounds:** {bounds}"
    )


# ============================================================
# NORMALIZATION
# ============================================================

with st.spinner(
    "Preparing satellite data..."
):

    normalized = normalize_image(
        image
    )


# ============================================================
# RGB
# ============================================================

rgb = create_rgb(
    normalized
)


st.header(
    "🛰️ Satellite Image"
)

st.image(
    rgb,
    caption="Sentinel-2 True Color Composite",
    use_container_width=True
)


# ============================================================
# ANALYZE
# ============================================================

analyze = st.button(
    "🔍 ANALYZE WITH OCEANMIND AI",
    type="primary",
    use_container_width=True
)


if analyze:

    st.divider()

    st.header(
        "🤖 OceanMind AI Analysis"
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    with st.spinner(
        "Loading AI model..."
    ):

        model = load_model()


    # ========================================================
    # PREPARE INPUT
    # ========================================================

    tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    )


    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        mode="bilinear",
        align_corners=False
    )


    tensor = tensor.to(
        device
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    with st.spinner(
        "OceanMind AI is analyzing marine pollution..."
    ):

        with torch.no_grad():

            output = model(
                tensor
            )

            probability = torch.sigmoid(
                output
            )


    # ========================================================
    # MASK
    # ========================================================

    low_mask = (
        probability[0, 0]
        .cpu()
        .numpy()
        >= THRESHOLD
    )


    # ========================================================
    # RESIZE MASK
    # ========================================================

    mask_image = Image.fromarray(
        (
            low_mask.astype(
                np.uint8
            ) * 255
        )
    )


    mask_image = mask_image.resize(
        (
            width,
            height
        ),
        Image.Resampling.NEAREST
    )


    full_mask = (
        np.array(mask_image)
        > 127
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    debris_pixels = int(
        np.sum(full_mask)
    )

    total_pixels = int(
        full_mask.size
    )

    debris_percentage = (
        debris_pixels /
        max(
            1,
            total_pixels
        )
    ) * 100


    # ========================================================
    # CONFIDENCE
    # ========================================================

    probability_map = (
        probability[0, 0]
        .cpu()
        .numpy()
    )


    if np.any(low_mask):

        confidence = (
            np.mean(
                probability_map[
                    low_mask
                ]
            ) * 100
        )

    else:

        confidence = (
            np.max(
                probability_map
            ) * 100
        )


    confidence = float(
        np.clip(
            confidence,
            0,
            100
        )
    )


    # ========================================================
    # SEVERITY
    # ========================================================

    severity, severity_description = (
        get_severity(
            debris_percentage
        )
    )


    # ========================================================
    # OVERLAY
    # ========================================================

    overlay = create_overlay(
        rgb,
        full_mask
    )


    # ========================================================
    # MAIN RESULT
    # ========================================================

    st.header(
        "📋 Detection Result"
    )


    if debris_pixels > 0:

        st.warning(
            "⚠️ Potential marine debris detected"
        )

    else:

        st.success(
            "✅ No potential marine debris detected"
        )


    # ========================================================
    # METRICS
    # ========================================================

    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.metric(
            "Detected Area",
            f"{debris_percentage:.2f}%"
        )


    with m2:

        st.metric(
            "Debris Pixels",
            f"{debris_pixels:,}"
        )


    with m3:

        st.metric(
            "AI Confidence",
            f"{confidence:.1f}%"
        )


    with m4:

        st.metric(
            "Severity",
            severity
        )


    st.write(
        f"**Assessment:** {severity_description}"
    )


    # ========================================================
    # VISUAL RESULTS
    # ========================================================

    st.header(
        "🎯 Segmentation Results"
    )


    v1, v2 = st.columns(2)


    with v1:

        st.image(
            full_mask,
            caption="Predicted Marine Debris Mask",
            use_container_width=True
        )


    with v2:

        st.image(
            overlay,
            caption="Marine Debris Detection Overlay",
            use_container_width=True
        )


    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    st.header(
        "📥 Export Results"
    )


    mask_bytes = array_to_png_bytes(
        full_mask,
        is_mask=True
    )


    overlay_bytes = array_to_png_bytes(
        overlay,
        is_mask=False
    )


    report = f"""
OCEANMIND AI ANALYSIS REPORT
========================================

Input File:
{uploaded_file.name}

Model:
U-Net CNN

Input:
11-band Sentinel-2

Processing Size:
128 x 128

Detection Threshold:
{THRESHOLD:.2f}

Total Pixels:
{total_pixels:,}

Detected Debris Pixels:
{debris_pixels:,}

Detected Area:
{debris_percentage:.2f}%

AI Confidence Estimate:
{confidence:.1f}%

Pollution Severity:
{severity}

Assessment:
{severity_description}

Coordinate Reference System:
{crs}

========================================

NOTE:
OceanMind AI is a research prototype.
Detected regions represent potential marine
debris and should be independently validated
before operational environmental decisions.
"""


    d1, d2, d3 = st.columns(3)


    with d1:

        st.download_button(
            "⬇️ Download Mask",
            data=mask_bytes,
            file_name="oceanmind_debris_mask.png",
            mime="image/png",
            use_container_width=True
        )


    with d2:

        st.download_button(
            "⬇️ Download Overlay",
            data=overlay_bytes,
            file_name="oceanmind_debris_overlay.png",
            mime="image/png",
            use_container_width=True
        )


    with d3:

        st.download_button(
            "📄 Download Report",
            data=report,
            file_name="oceanmind_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )


    # ========================================================
    # AI INFORMATION
    # ========================================================

    st.divider()

    st.header(
        "🧠 AI Model Information"
    )


    a1, a2, a3 = st.columns(3)


    with a1:

        st.write(
            "**Deep Learning Architecture**"
        )

        st.write(
            "U-Net Convolutional Neural Network"
        )


    with a2:

        st.write(
            "**Input Data**"
        )

        st.write(
            "11-band Sentinel-2 imagery"
        )


    with a3:

        st.write(
            "**AI Task**"
        )

        st.write(
            "Pixel-level marine debris segmentation"
        )


    # ========================================================
    # PIPELINE
    # ========================================================

    st.header(
        "🔄 Analysis Pipeline"
    )

    st.success(
        "Satellite Image"
    )

    st.write("↓")

    st.success(
        "11-Band Preprocessing"
    )

    st.write("↓")

    st.success(
        "U-Net Deep Learning"
    )

    st.write("↓")

    st.success(
        "Pixel Segmentation"
    )

    st.write("↓")

    st.success(
        "Marine Debris Detection"
    )

    st.write("↓")

    st.success(
        "Pollution Severity Assessment"
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.info(
        "⚠️ OceanMind AI is currently a research "
        "prototype. The AI confidence shown here is "
        "derived from model prediction probabilities "
        "and should not be interpreted as a validated "
        "real-world accuracy measure."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌊 OceanMind AI | AI-Based Marine Pollution Detection"
) 
