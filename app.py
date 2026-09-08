import os
import tempfile
from io import BytesIO

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

MODEL_PATH = "marine_debris_segmentation.pth"

EXPECTED_BANDS = 11

INPUT_SIZE = 128

THRESHOLD = 0.50


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .severity-low {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        background-color: #dff5e1;
        color: #176b2c;
        font-size: 24px;
        font-weight: 700;
    }

    .severity-moderate {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        background-color: #fff1c7;
        color: #8a6200;
        font-size: 24px;
        font-weight: 700;
    }

    .severity-high {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        background-color: #ffdede;
        color: #a00000;
        font-size: 24px;
        font-weight: 700;
    }

    .severity-none {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        background-color: #e5e7eb;
        color: #374151;
        font-size: 24px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


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

        # Bottleneck

        self.bottleneck = DoubleConv(
            128,
            256
        )

        # Decoder

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        # Output

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # Bottleneck

        b = self.bottleneck(
            self.pool(e3)
        )

        # Decoder 3

        d3 = self.up3(b)

        if d3.shape[-2:] != e3.shape[-2:]:

            d3 = F.interpolate(
                d3,
                size=e3.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

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

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

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

    if not os.path.exists(MODEL_PATH):

        return None, (
            f"Model file not found: {MODEL_PATH}"
        )

    try:

        device = torch.device("cpu")

        model = UNet(
            input_channels=EXPECTED_BANDS
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False
        )

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                state_dict = checkpoint[
                    "model_state_dict"
                ]

            elif "state_dict" in checkpoint:

                state_dict = checkpoint[
                    "state_dict"
                ]

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint

        model.load_state_dict(
            state_dict
        )

        model.to(device)

        model.eval()

        return model, None

    except Exception as e:

        return None, str(e)


# ============================================================
# NORMALIZE IMAGE
# ============================================================

def normalize_image(image):

    image = image.astype(
        np.float32
    )

    normalized = np.zeros_like(
        image,
        dtype=np.float32
    )

    for i in range(
        image.shape[0]
    ):

        band = image[i]

        band = np.nan_to_num(
            band,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        minimum = np.min(band)

        maximum = np.max(band)

        if maximum > minimum:

            normalized[i] = (
                band - minimum
            ) / (
                maximum - minimum
            )

        else:

            normalized[i] = 0.0

    return normalized


# ============================================================
# TRUE COLOR
# Sentinel-2:
# B4 = Red
# B3 = Green
# B2 = Blue
# ============================================================

def create_true_color(image):

    if image.shape[0] < 4:

        return None

    red = image[3].astype(
        np.float32
    )

    green = image[2].astype(
        np.float32
    )

    blue = image[1].astype(
        np.float32
    )

    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    low = np.percentile(
        rgb,
        2
    )

    high = np.percentile(
        rgb,
        98
    )

    if high > low:

        rgb = (
            rgb - low
        ) / (
            high - low
        )

    rgb = np.clip(
        rgb,
        0,
        1
    )

    return (
        rgb * 255
    ).astype(
        np.uint8
    )


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


def get_severity_class(severity):

    if severity == "Low":

        return "severity-low"

    elif severity == "Moderate":

        return "severity-moderate"

    elif severity == "High":

        return "severity-high"

    else:

        return "severity-none"


# ============================================================
# MASK IMAGE
# ============================================================

def create_mask_image(mask):

    mask_uint8 = (
        mask.astype(
            np.uint8
        ) * 255
    )

    return Image.fromarray(
        mask_uint8,
        mode="L"
    )


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    rgb,
    mask
):

    base = Image.fromarray(
        rgb
    ).convert(
        "RGBA"
    )

    overlay = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            4
        ),
        dtype=np.uint8
    )

    overlay[
        mask == 1
    ] = [
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

    return result.convert(
        "RGB"
    )


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    filename,
    width,
    height,
    bands,
    crs,
    bounds,
    latitude,
    longitude,
    debris_pixels,
    total_pixels,
    area_percentage,
    confidence,
    severity,
    threshold,
    max_probability
):

    from reportlab.lib.pagesizes import A4

    from reportlab.lib import colors

    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle
    )

    from reportlab.lib.enums import TA_CENTER

    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "OceanMindTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=24,
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        "OceanMindSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=12,
        spaceAfter=20
    )

    story = []

    # Title

    story.append(
        Paragraph(
            "OceanMind AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Based Marine Pollution Detection Report",
            subtitle_style
        )
    )

    # Satellite information

    story.append(
        Paragraph(
            "1. Satellite Image Information",
            styles["Heading2"]
        )
    )

    image_data = [

        [
            "Parameter",
            "Value"
        ],

        [
            "Filename",
            str(filename)
        ],

        [
            "Spectral Bands",
            str(bands)
        ],

        [
            "Image Width",
            str(width)
        ],

        [
            "Image Height",
            str(height)
        ],

        [
            "CRS",
            str(crs)
        ],

        [
            "Bounds",
            str(bounds)
        ]

    ]

    table = Table(
        image_data,
        colWidths=[
            150,
            330
        ]
    )

    table.setStyle(
        TableStyle(
            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )

            ]
        )
    )

    story.append(table)

    story.append(
        Spacer(
            1,
            20
        )
    )

    # Detection results

    story.append(
        Paragraph(
            "2. AI Detection Results",
            styles["Heading2"]
        )
    )

    detection_data = [

        [
            "Metric",
            "Result"
        ],

        [
            "Debris Pixels",
            f"{debris_pixels:,}"
        ],

        [
            "Total Pixels",
            f"{total_pixels:,}"
        ],

        [
            "Detected Area",
            f"{area_percentage:.2f}%"
        ],

        [
            "AI Confidence Estimate",
            f"{confidence:.1f}%"
        ],

        [
            "Maximum Probability",
            f"{max_probability * 100:.1f}%"
        ],

        [
            "Detection Threshold",
            str(threshold)
        ],

        [
            "Severity",
            severity
        ]

    ]

    table2 = Table(
        detection_data,
        colWidths=[
            250,
            230
        ]
    )

    table2.setStyle(
        TableStyle(
            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )

            ]
        )
    )

    story.append(table2)

    story.append(
        Spacer(
            1,
            20
        )
    )

    # Geographic information

    story.append(
        Paragraph(
            "3. Geographic Information",
            styles["Heading2"]
        )
    )

    geographic_data = [

        [
            "Parameter",
            "Value"
        ],

        [
            "Latitude",
            f"{latitude:.6f}"
        ],

        [
            "Longitude",
            f"{longitude:.6f}"
        ]

    ]

    table3 = Table(
        geographic_data,
        colWidths=[
            150,
            330
        ]
    )

    table3.setStyle(
        TableStyle(
            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )

            ]
        )
    )

    story.append(table3)

    story.append(
        Spacer(
            1,
            20
        )
    )

    # AI model

    story.append(
        Paragraph(
            "4. AI Model",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            "Architecture: U-Net Convolutional Neural Network<br/>"
            "Input: 11-band Sentinel-2 satellite imagery<br/>"
            "AI Input Size: 128 × 128 pixels<br/>"
            "Output: Pixel-wise marine debris segmentation mask",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(
            1,
            20
        )
    )

    # Interpretation

    story.append(
        Paragraph(
            "5. Interpretation",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            "The detected area represents pixels classified "
            "by the model as potential marine debris. "
            "The AI confidence value is a model probability "
            "estimate and should not be interpreted as "
            "validated real-world accuracy.",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(
            1,
            20
        )
    )

    story.append(
        Paragraph(
            "OceanMind AI Research Prototype<br/>"
            "Results should be validated before real-world "
            "environmental decisions.",
            styles["Normal"]
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SESSION HISTORY
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌊 OceanMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Marine Pollution Detection & Monitoring'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Detect potential marine debris from "
    "11-band Sentinel-2 satellite imagery "
    "using a U-Net deep learning model."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🌊 OceanMind AI"
    )

    st.write(
        "AI-powered satellite analysis "
        "for marine pollution monitoring."
    )

    st.divider()

    st.subheader(
        "🤖 AI Model"
    )

    st.write(
        "**Architecture:** U-Net CNN"
    )

    st.write(
        "**Input Bands:** 11"
    )

    st.write(
        "**Input Size:** 128 × 128"
    )

    st.write(
        "**Output:** Segmentation Mask"
    )

    st.write(
        "**Threshold:** 0.50"
    )

    st.write(
        "**Device:** CPU"
    )

    st.divider()

    st.subheader(
        "📊 Severity Levels"
    )

    st.write(
        "⚪ No Detection"
    )

    st.write(
        "🟢 Low"
    )

    st.write(
        "🟡 Moderate"
    )

    st.write(
        "🔴 High"
    )

    st.divider()

    st.subheader(
        "Project Status"
    )

    st.success(
        "Model Ready"
    )

    st.success(
        "Dashboard Ready"
    )

    st.success(
        "Cloud Deployment Ready"
    )

    st.divider()

    st.caption(
        "OceanMind AI is a research prototype."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.header(
    "📡 Satellite Image Analysis"
)

uploaded_file = st.file_uploader(
    "Upload an 11-band Sentinel-2 GeoTIFF",
    type=[
        "tif",
        "tiff"
    ]
)


# ============================================================
# APPLICATION
# ============================================================

if uploaded_file is not None:

    temp_path = None

    try:

        # ----------------------------------------------------
        # SAVE TEMPORARY FILE
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".tif"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        # ----------------------------------------------------
        # READ GEOTIFF
        # ----------------------------------------------------

        with rasterio.open(
            temp_path
        ) as src:

            image = src.read()

            crs = src.crs

            bounds = src.bounds

            width = src.width

            height = src.height

            band_count = src.count

            dtype = src.dtypes[0]

            resolution = src.res

        # ----------------------------------------------------
        # VALIDATE BAND COUNT
        # ----------------------------------------------------

        if band_count != EXPECTED_BANDS:

            st.error(
                f"Expected {EXPECTED_BANDS} bands, "
                f"but this image contains {band_count} bands."
            )

            st.stop()

        st.success(
            "✅ Valid 11-band Sentinel-2 GeoTIFF detected."
        )

        # ----------------------------------------------------
        # SATELLITE METADATA
        # ----------------------------------------------------

        st.header(
            "🛰️ Satellite Metadata"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Spectral Bands",
            band_count
        )

        c2.metric(
            "Width",
            width
        )

        c3.metric(
            "Height",
            height
        )

        c4.metric(
            "Data Type",
            dtype
        )

        with st.expander(
            "🌍 Complete Geospatial Metadata"
        ):

            m1, m2 = st.columns(2)

            with m1:

                st.write(
                    f"**CRS:** {crs}"
                )

                st.write(
                    f"**Resolution:** {resolution}"
                )

                st.write(
                    f"**Width:** {width}"
                )

                st.write(
                    f"**Height:** {height}"
                )

            with m2:

                st.write(
                    f"**Left:** {bounds.left}"
                )

                st.write(
                    f"**Right:** {bounds.right}"
                )

                st.write(
                    f"**Top:** {bounds.top}"
                )

                st.write(
                    f"**Bottom:** {bounds.bottom}"
                )

        # ----------------------------------------------------
        # TRUE COLOR IMAGE
        # ----------------------------------------------------

        rgb = create_true_color(
            image
        )

        st.header(
            "🖼️ Satellite Image"
        )

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

            # ------------------------------------------------
            # LOAD MODEL
            # ------------------------------------------------

            model, model_error = load_model()

            if model_error is not None:

                st.error(
                    f"Model loading failed: {model_error}"
                )

                st.stop()

            # ------------------------------------------------
            # NORMALIZATION
            # ------------------------------------------------

            with st.spinner(
                "🔧 Preprocessing satellite imagery..."
            ):

                normalized = normalize_image(
                    image
                )

                tensor = torch.from_numpy(
                    normalized
                ).float()

                tensor = tensor.unsqueeze(0)

                tensor = F.interpolate(
                    tensor,
                    size=(
                        INPUT_SIZE,
                        INPUT_SIZE
                    ),
                    mode="bilinear",
                    align_corners=False
                )

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            with st.spinner(
                "🤖 OceanMind AI is analyzing the image..."
            ):

                with torch.no_grad():

                    output = model(
                        tensor
                    )

                    probability = torch.sigmoid(
                        output
                    )

            probability = (
                probability
                .squeeze()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # RESIZE TO ORIGINAL IMAGE
            # ------------------------------------------------

            probability_original = np.array(

                Image.fromarray(
                    probability.astype(
                        np.float32
                    ),
                    mode="F"
                ).resize(
                    (
                        width,
                        height
                    ),
                    Image.Resampling.BILINEAR
                )

            )

            # ------------------------------------------------
            # CREATE MASK
            # ------------------------------------------------

            mask = (
                probability_original
                >= THRESHOLD
            ).astype(
                np.uint8
            )

            # ------------------------------------------------
            # CALCULATE STATISTICS
            # ------------------------------------------------

            debris_pixels = int(
                np.sum(mask)
            )

            total_pixels = int(
                mask.size
            )

            area_percentage = (
                debris_pixels /
                total_pixels
            ) * 100

            severity = get_severity(
                area_percentage
            )

            max_probability = float(
                np.max(
                    probability_original
                )
            )

            detected_probabilities = (
                probability_original[
                    mask == 1
                ]
            )

            if len(
                detected_probabilities
            ) > 0:

                confidence = (
                    float(
                        np.mean(
                            detected_probabilities
                        )
                    ) * 100
                )

            else:

                confidence = (
                    max_probability * 100
                )

            # ------------------------------------------------
            # GEOGRAPHIC COORDINATES
            # ------------------------------------------------

            try:

                geographic_bounds = transform_bounds(
                    crs,
                    "EPSG:4326",
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top
                )

                lon_min = geographic_bounds[0]

                lat_min = geographic_bounds[1]

                lon_max = geographic_bounds[2]

                lat_max = geographic_bounds[3]

                center_lon = (
                    lon_min +
                    lon_max
                ) / 2

                center_lat = (
                    lat_min +
                    lat_max
                ) / 2

                geographic_available = True

            except Exception:

                center_lon = 0.0

                center_lat = 0.0

                geographic_available = False

            # ------------------------------------------------
            # SAVE HISTORY
            # ------------------------------------------------

            history_record = {

                "Image":
                    uploaded_file.name,

                "Detected Area (%)":
                    round(
                        area_percentage,
                        4
                    ),

                "Debris Pixels":
                    debris_pixels,

                "Confidence (%)":
                    round(
                        confidence,
                        2
                    ),

                "Severity":
                    severity

            }

            st.session_state.history.append(
                history_record
            )

            # ------------------------------------------------
            # DETECTION RESULT
            # ------------------------------------------------

            st.header(
                "🚨 Detection Result"
            )

            if debris_pixels > 0:

                st.warning(
                    "Potential marine debris detected."
                )

            else:

                st.success(
                    "No potential marine debris detected."
                )

            # ------------------------------------------------
            # MAIN METRICS
            # ------------------------------------------------

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Detected Area",
                f"{area_percentage:.2f}%"
            )

            c2.metric(
                "Debris Pixels",
                f"{debris_pixels:,}"
            )

            c3.metric(
                "AI Confidence",
                f"{confidence:.1f}%"
            )

            c4.metric(
                "Maximum Probability",
                f"{max_probability * 100:.1f}%"
            )

            # ------------------------------------------------
            # SEVERITY
            # ------------------------------------------------

            st.subheader(
                "⚠️ Pollution Severity"
            )

            st.markdown(
                f"""
                <div class="{get_severity_class(severity)}">
                    {severity}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # POLLUTION STATISTICS
            # ------------------------------------------------

            st.header(
                "📊 Pollution Statistics"
            )

            s1, s2 = st.columns(2)

            with s1:

                st.write(
                    f"**Total image pixels:** "
                    f"{total_pixels:,}"
                )

                st.write(
                    f"**Potential debris pixels:** "
                    f"{debris_pixels:,}"
                )

                st.write(
                    f"**Detection threshold:** "
                    f"{THRESHOLD}"
                )

            with s2:

                st.write(
                    f"**Detected coverage:** "
                    f"{area_percentage:.4f}%"
                )

                st.write(
                    f"**AI confidence estimate:** "
                    f"{confidence:.2f}%"
                )

                st.write(
                    f"**Severity:** "
                    f"{severity}"
                )

            # ------------------------------------------------
            # COVERAGE VISUALIZATION
            # ------------------------------------------------

            coverage_value = min(
                area_percentage / 1.0,
                1.0
            )

            st.progress(
                coverage_value
            )

            st.caption(
                "Coverage bar is normalized to 1% "
                "detected area."
            )

            # ------------------------------------------------
            # CONFIDENCE WARNING
            # ------------------------------------------------

            st.info(
                "AI Confidence is the model's predicted "
                "probability for detected pixels. "
                "It is not validated real-world accuracy."
            )

            # ------------------------------------------------
            # MAP
            # ------------------------------------------------

            st.header(
                "🗺️ Geographic Location"
            )

            if geographic_available:

                map_data = pd.DataFrame(
                    {
                        "latitude": [
                            center_lat
                        ],

                        "longitude": [
                            center_lon
                        ]
                    }
                )

                st.map(
                    map_data,
                    latitude="latitude",
                    longitude="longitude",
                    zoom=8
                )

                map1, map2 = st.columns(2)

                with map1:

                    st.metric(
                        "Latitude",
                        f"{center_lat:.6f}"
                    )

                with map2:

                    st.metric(
                        "Longitude",
                        f"{center_lon:.6f}"
                    )

                st.caption(
                    "The map displays the approximate "
                    "geographic center of the satellite patch."
                )

            else:

                st.warning(
                    "The uploaded GeoTIFF does not provide "
                    "a geographic CRS that could be converted "
                    "to latitude/longitude."
                )

            # ------------------------------------------------
            # SEGMENTATION
            # ------------------------------------------------

            st.header(
                "📊 Segmentation Results"
            )

            mask_image = create_mask_image(
                mask
            )

            overlay_image = create_overlay(
                rgb,
                mask
            )

            r1, r2 = st.columns(2)

            with r1:

                st.image(
                    mask_image,
                    caption="Marine Debris Segmentation Mask",
                    use_container_width=True
                )

            with r2:

                st.image(
                    overlay_image,
                    caption="Marine Debris Detection Overlay",
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

            # ------------------------------------------------
            # DOWNLOAD OVERLAY
            # ------------------------------------------------

            overlay_buffer = BytesIO()

            overlay_image.save(
                overlay_buffer,
                format="PNG"
            )

            d1, d2 = st.columns(2)

            with d1:

                st.download_button(
                    "⬇️ Download Segmentation Mask",
                    data=mask_buffer.getvalue(),
                    file_name="oceanmind_debris_mask.png",
                    mime="image/png",
                    use_container_width=True
                )

            with d2:

                st.download_button(
                    "⬇️ Download Detection Overlay",
                    data=overlay_buffer.getvalue(),
                    file_name="oceanmind_detection_overlay.png",
                    mime="image/png",
                    use_container_width=True
                )

            # ------------------------------------------------
            # PDF REPORT
            # ------------------------------------------------

            st.header(
                "📄 Professional Analysis Report"
            )

            if geographic_available:

                pdf_latitude = center_lat

                pdf_longitude = center_lon

            else:

                pdf_latitude = 0.0

                pdf_longitude = 0.0

            try:

                pdf_data = create_pdf_report(

                    filename=uploaded_file.name,

                    width=width,

                    height=height,

                    bands=band_count,

                    crs=crs,

                    bounds=bounds,

                    latitude=pdf_latitude,

                    longitude=pdf_longitude,

                    debris_pixels=debris_pixels,

                    total_pixels=total_pixels,

                    area_percentage=area_percentage,

                    confidence=confidence,

                    severity=severity,

                    threshold=THRESHOLD,

                    max_probability=max_probability

                )

                st.download_button(

                    "📄 Download Professional PDF Report",

                    data=pdf_data,

                    file_name="OceanMind_AI_Report.pdf",

                    mime="application/pdf",

                    use_container_width=True

                )

            except Exception as e:

                st.error(
                    f"PDF generation error: {e}"
                )

            # ------------------------------------------------
            # AI PIPELINE
            # ------------------------------------------------

            st.divider()

            st.header(
                "⚙️ AI Analysis Pipeline"
            )

            pipeline = [

                "🛰️ 11-band Sentinel-2 GeoTIFF",

                "⬇️",

                "🔧 Band-wise normalization",

                "⬇️",

                "📐 Resize to 128 × 128",

                "⬇️",

                "🧠 U-Net CNN",

                "⬇️",

                "🎯 Pixel-wise probability map",

                "⬇️",

                "✂️ Threshold = 0.50",

                "⬇️",

                "🚨 Marine debris segmentation",

                "⬇️",

                "📊 Pollution statistics",

                "⬇️",

                "⚠️ Severity classification",

                "⬇️",

                "🗺️ Geographic visualization",

                "⬇️",

                "📄 Professional PDF report"

            ]

            for step in pipeline:

                st.write(step)

    except Exception as e:

        st.error(
            f"Error processing the satellite image: {e}"
        )

    finally:

        if temp_path is not None:

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


# ============================================================
# DETECTION HISTORY
# ============================================================

if len(
    st.session_state.history
) > 0:

    st.divider()

    st.header(
        "📜 Detection History"
    )

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "🗑️ Clear Detection History"
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌊 OceanMind AI | AI-Based Marine Pollution "
    "Detection using Sentinel-2 Satellite Imagery"
)

st.caption(
    "Research prototype — results should be validated "
    "before real-world environmental decisions."
)
