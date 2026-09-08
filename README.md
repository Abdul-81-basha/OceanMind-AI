# 🌊 OceanMind AI
**Live Demo:**  
https://oceanmind-ai-eoupyhve7q2ildamdroszq.streamlit.app/

## AI-Based Marine Pollution Detection & Monitoring

OceanMind AI is an artificial intelligence system designed to detect and monitor marine pollution, particularly floating marine debris, using multispectral Sentinel-2 satellite imagery.

The system uses an 11-band satellite image as input and applies a U-Net deep learning model to generate a pixel-level marine debris detection mask.

---

## 🎯 Problem Statement

Marine pollution caused by floating debris is difficult to monitor over large ocean and coastal areas using conventional ground-based methods.

OceanMind AI provides an automated satellite-based approach for detecting potential marine debris and visualizing detected pollution areas.

---

## 🚀 Key Features

- 🛰️ Sentinel-2 multispectral satellite image analysis
- 🌈 11-band satellite image processing
- 🤖 U-Net deep learning segmentation
- 🔍 Pixel-level marine debris detection
- 📊 Pollution area estimation
- 🗺️ Geographic scene visualization
- 📈 Detection severity classification
- 🖼️ Prediction mask and overlay visualization
- 📄 Automatic detection report
- ⬇️ Downloadable prediction results
- 🌐 Streamlit web dashboard

---

## 🧠 AI Model

The project uses a **U-Net Convolutional Neural Network (CNN)** for semantic segmentation.

### Architecture

```text
11-Band Sentinel-2 Image
          ↓
       Encoder
          ↓
      Bottleneck
          ↓
       Decoder
          ↓
   Skip Connections
          ↓
  1-Channel Prediction
          ↓
 Marine Debris Mask
