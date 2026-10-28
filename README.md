# OceanMind AI 2.0 — Frontend

Next.js frontend scaffold for a U-Net multi-class marine-pollution segmentation platform.

## Requirements
- Node.js 20.9+ recommended
- A running OceanMind FastAPI backend for real inference

## Run locally
1. Open a terminal in this folder.
2. Install dependencies: `npm install`
3. Copy `.env.example` to `.env.local`.
4. Keep `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` for local FastAPI.
5. Start the frontend: `npm run dev`
6. Open http://localhost:3000.

## Backend contract expected by the frontend
`POST /predict` as multipart/form-data:
- `file`: uploaded image
- `sensor_type`: `optical` or `sar`

Expected JSON response schema example (not a real prediction):
```json
{
  "pollution_type": "Marine debris",
  "severity": "Low",
  "pollution_area_percent": 0.42,
  "confidence": 0.81,
  "dice": 0.67,
  "class_percentages": {
    "Marine debris": 0.42,
    "Oil spill": 0.0,
    "Other pollution": 0.0,
    "Background": 99.58
  },
  "mask_url": "/results/example-mask.png",
  "overlay_url": "/results/example-overlay.png"
}
```
The backend must return real model outputs. This frontend does not run inference in the browser. GeoTIFF decoding and U-Net inference belong in the backend. FastAPI must enable CORS for your Vercel origin.

## Deploy to Vercel
1. Push this folder to GitHub.
2. Import the repository in Vercel.
3. Set Root Directory to the folder containing this package.json.
4. Set `NEXT_PUBLIC_API_URL` to your deployed Render API URL.
5. Deploy.

Keep large datasets and model checkpoints out of GitHub.
