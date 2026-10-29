
"use client";

import { ChangeEvent, DragEvent, useState } from "react";
import {
  Activity, ArrowDownToLine, ArrowRight, CheckCircle2,
  CloudUpload, FileImage, Globe2, Layers3, LoaderCircle,
  MapPinned, Menu, Microscope, ShieldCheck, Waves, X,
  Gauge, Droplets
} from "lucide-react";

type Result = {
  pollution_type?: string;
  detected_class?: string;
  severity?: string;
  pollution_area_percent?: number;
  confidence?: number;
  dice?: number;
  class_percentages?: Record<string, number>;
  mask_url?: string | null;
  overlay_url?: string | null;
  output_image_url?: string | null;
  processed_image_url?: string | null;
  message?: string;
  sensor_type?: string;
};

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const colors: Record<string, string> = {
  "Marine Debris": "#24c7b5",
  "Marine debris": "#24c7b5",
  "Dense Sargassum": "#ffb454",
  "Sparse Sargassum": "#e5b65c",
  "Natural Organic Material": "#b6a0ff",
  Ship: "#e879b9",
  Clouds: "#cbd5e1",
  "Marine Water": "#78909c",
  "Sediment-Laden Water": "#5486c7",
  Foam: "#e5e7eb",
  "Turbid Water": "#3da6b3",
  "Shallow Water": "#57a77b",
  Waves: "#7bc9d4",
  "Cloud Shadows": "#7b8294",
  Wakes: "#8bcfba",
  "Mixed Water": "#b7a0d9"
};

function imageUrl(url?: string | null) {
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }
  return `${API_URL}${url.startsWith("/") ? "" : "/"}${url}`;
}

export default function Home() {
  const [sensor, setSensor] = useState("optical");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [zoomImage, setZoomImage] = useState<{ title: string; url: string } | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [menu, setMenu] = useState(false);

  function acceptFile(f?: File) {
    if (!f) return;

    const name = f.name.toLowerCase();
    const allowed = [".tif", ".tiff", ".png", ".jpg", ".jpeg"];

    if (!allowed.some((ext) => name.endsWith(ext))) {
      setError("Choose a GeoTIFF, TIFF, PNG, or JPEG image.");
      return;
    }

    setFile(f);
    setError("");
    setResult(null);
  }

  function fileChange(e: ChangeEvent<HTMLInputElement>) {
    acceptFile(e.target.files?.[0]);
  }

  function drop(e: DragEvent<HTMLLabelElement>) {
    e.preventDefault();
    setDragging(false);
    acceptFile(e.dataTransfer.files?.[0]);
  }
