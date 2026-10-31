"use client";

import { ChangeEvent, DragEvent, useMemo, useState } from "react";
import {
  Activity, ArrowDownToLine, ArrowRight, CheckCircle2, CloudUpload,
  FileImage, Globe2, Layers3, LoaderCircle, MapPinned, Menu, Microscope,
  ShieldCheck, Waves, X, Gauge, Droplets, Satellite, BarChart3, FileText,
  Moon, Sun, ZoomIn, RefreshCw, Sparkles, Info, ChevronRight
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
  "Marine Debris": "#26d9c2",
  "Marine debris": "#26d9c2",
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
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
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
  const [lightMode, setLightMode] = useState(false);
  const [activeSection, setActiveSection] = useState("dashboard");

  function acceptFile(f?: File) {
    if (!f) return;
    const name = f.name.toLowerCase();
    const allowed = [".tif", ".tiff", ".png", ".jpg", ".jpeg"];
    if (!allowed.some((ext) => name.endsWith(ext))) {
      setError("Choose a GeoTIFF, TIFF, PNG, or JPEG image.");
      return;
    }
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
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

  async function analyze() {
    if (!file) {
      setError("Upload a satellite image first.");
      return;
    }
    if (sensor === "sar") {
      setError("The current U-Net was trained on 11-band optical imagery. SAR analysis is not supported by this checkpoint.");
      return;
    }
    const name = file.name.toLowerCase();
    if (!name.endsWith(".tif") && !name.endsWith(".tiff")) {
      setError("The current model requires an 11-band, 256 × 256 optical GeoTIFF. PNG and JPEG files cannot be used directly.");
      return;
    }

    setBusy(true);
    setError("");
    setResult(null);
    const form = new FormData();
    form.append("file", file);
    form.append("sensor_type", sensor);

    try {
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || data.message || `Analysis failed (${response.status}).`);
      }
      setResult(data as Result);
      setActiveSection("results");
      window.setTimeout(() => document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }), 100);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unable to connect to API.";
      setError(msg.includes("Failed to fetch")
        ? `Cannot reach backend at ${API_URL}. Check the API service and CORS configuration.`
        : msg);
    } finally {
      setBusy(false);
    }
  }

  async function download(url?: string | null, name = "oceanmind-result.png") {
    if (!url) return;
    try {
      const response = await fetch(imageUrl(url));
      if (!response.ok) throw new Error(`Download failed with HTTP ${response.status}.`);
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(blobUrl), 5000);
    } catch (e) {
      console.error("OceanMind download error:", e);
      setError(e instanceof Error
        ? `${e.message} If this is a CORS error, check the backend CORS settings.`
        : "Unable to download the output image.");
    }
  }

  function downloadReport() {
    if (!result) return;
    const rows = Object.entries(result.class_percentages || {})
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => `<tr><td>${name}</td><td>${value.toFixed(2)}%</td></tr>`).join("");
    const report = `<!doctype html><html><head><meta charset="utf-8"><title>OceanMind AI Report</title>
    <style>body{font-family:Arial,sans-serif;color:#102c38;padding:36px;max-width:850px;margin:auto}h1{color:#087f83}table{border-collapse:collapse;width:100%;margin-top:20px}td,th{padding:12px;border-bottom:1px solid #d9e6e7;text-align:left}.metric{display:inline-block;padding:16px;margin:8px;background:#eff8f7;border-radius:12px}</style></head><body>
    <h1>OceanMind AI — Analysis Report</h1><p>Marine pollution segmentation research prototype</p>
    <div class="metric"><b>Predicted class</b><br>${result.pollution_type || result.detected_class || "Not provided"}</div>
    <div class="metric"><b>Severity</b><br>${result.severity || "Not provided"}</div>
    <div class="metric"><b>Predicted area</b><br>${typeof result.pollution_area_percent === "number" ? result.pollution_area_percent.toFixed(3) + "%" : "Not provided"}</div>
    <div class="metric"><b>Mean max probability</b><br>${typeof result.confidence === "number" ? (result.confidence <= 1 ? result.confidence * 100 : result.confidence).toFixed(1) + "%" : "Not provided"}</div>
    <h2>Class breakdown</h2><table><tr><th>Class</th><th>Predicted share</th></tr>${rows || "<tr><td colspan='2'>No class breakdown returned</td></tr>"}</table>
    <p><b>Note:</b> Experimental predictions require validation against labeled data and expert review. Mean maximum probability is not validated accuracy.</p></body></html>`;
    const blob = new Blob([report], { type: "text/html;charset=utf-8" });
    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = blobUrl;
    link.download = "oceanmind-analysis-report.html";
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(blobUrl), 5000);
  }

  const detected = result?.pollution_type || result?.detected_class || "Analysis complete";
  const area = typeof result?.pollution_area_percent === "number" ? `${result.pollution_area_percent.toFixed(3)}%` : "—";
  const confidence = typeof result?.confidence === "number"
    ? `${(result.confidence <= 1 ? result.confidence * 100 : result.confidence).toFixed(1)}%`
    : "Not provided";
  const maskUrl = result?.mask_url;
  const overlayUrl = result?.overlay_url || result?.output_image_url;
  const originalUrl = result?.processed_image_url || previewUrl;
  const imagePanels = [
    { title: "Original image", url: originalUrl, alt: "Original satellite image" },
    { title: "Segmentation mask", url: maskUrl, alt: "Predicted segmentation mask" },
    { title: "Prediction overlay", url: overlayUrl, alt: "Color-coded prediction overlay" }
  ];
  const classRows = useMemo(() => Object.entries(result?.class_percentages || {}).sort((a, b) => b[1] - a[1]), [result]);
  const rootClass = lightMode ? "ocean-app light" : "ocean-app";

  return (
    <main className={rootClass}>
      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root{scroll-behavior:smooth}
        *{box-sizing:border-box}
        body{margin:0;background:#06131c}
        .ocean-app{--bg:#06131c;--bg2:#091c28;--panel:rgba(12,35,48,.88);--panel2:#102c3b;--line:rgba(87,190,195,.19);--text:#eaf8f8;--muted:#91adb5;--cyan:#42e4d0;--blue:#238fd0;--shadow:0 18px 55px rgba(0,0,0,.24);min-height:100vh;background:radial-gradient(ellipse at 78% 4%,rgba(0,151,164,.17),transparent 30%),radial-gradient(ellipse at 8% 30%,rgba(18,105,151,.13),transparent 28%),var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;overflow:hidden}
        .ocean-app.light{--bg:#edf6f8;--bg2:#e2f0f3;--panel:rgba(255,255,255,.91);--panel2:#fff;--line:rgba(24,113,137,.17);--text:#102e3a;--muted:#617d87;--cyan:#087f83;--blue:#247cc0;--shadow:0 18px 50px rgba(23,76,91,.10);background:radial-gradient(ellipse at 80% 0%,rgba(47,194,198,.18),transparent 30%),var(--bg)}
        .app-layout{display:grid;grid-template-columns:248px minmax(0,1fr);max-width:1600px;margin:auto;min-height:100vh}
        .sidebar{position:sticky;top:0;height:100vh;padding:25px 17px 20px;border-right:1px solid var(--line);background:linear-gradient(180deg,rgba(5,23,34,.97),rgba(7,30,42,.94));display:flex;flex-direction:column;z-index:20}
        .light .sidebar{background:linear-gradient(180deg,#f8fdff,#e8f4f7)}
        .brand{display:flex;align-items:center;gap:11px;padding:3px 8px 27px;text-decoration:none;color:var(--text)}
        .brandmark{width:42px;height:42px;border-radius:14px;display:grid;place-items:center;color:#06232d;background:linear-gradient(135deg,#5cf1d7,#299acb);box-shadow:0 0 25px rgba(66,228,208,.22)}
        .brandtext{font:800 19px 'Manrope',sans-serif;letter-spacing:-.7px;line-height:1.1}.brandtext span{color:var(--cyan)}.brandtext small{display:block;font:600 9px 'DM Sans',sans-serif;letter-spacing:1.7px;color:var(--muted);margin-top:5px}
        .side-label{padding:13px 12px 8px;color:#62818c;font-size:9px;font-weight:700;letter-spacing:1.8px}
        .side-nav{display:grid;gap:6px}.side-nav a{display:flex;align-items:center;gap:12px;padding:12px 13px;border-radius:11px;color:var(--muted);font-size:13px;font-weight:600;text-decoration:none;transition:.22s ease;border:1px solid transparent}.side-nav a:hover,.side-nav a.active{color:var(--text);background:linear-gradient(100deg,rgba(36,159,190,.2),rgba(40,122,170,.08));border-color:rgba(66,228,208,.2);box-shadow:inset 3px 0 0 var(--cyan)}.side-nav svg{width:17px;height:17px}
        .sidebar-bottom{margin-top:auto;padding:16px 13px;border:1px solid var(--line);border-radius:15px;background:linear-gradient(145deg,rgba(24,99,119,.24),rgba(9,38,53,.3));position:relative;overflow:hidden}.sidebar-bottom:after{content:'';position:absolute;width:120px;height:120px;border:1px solid rgba(66,228,208,.15);border-radius:50%;right:-45px;bottom:-72px;box-shadow:0 0 0 12px rgba(66,228,208,.035),0 0 0 25px rgba(66,228,208,.025)}.sidebar-bottom b{display:block;font:700 13px 'Manrope';margin-bottom:6px}.sidebar-bottom p{font-size:11px;line-height:1.55;color:var(--muted);margin:0}.side-status{display:flex;align-items:center;gap:7px;margin-top:14px;font-size:10px;color:var(--cyan)}.dot{width:7px;height:7px;border-radius:50%;background:#35dfad;box-shadow:0 0 10px #35dfad}
        .main-area{min-width:0;padding:0 34px 30px}.topbar{height:76px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);gap:20px}.top-title{font:700 14px 'Manrope';letter-spacing:.1px}.top-title small{display:block;color:var(--muted);font:400 10px 'DM Sans';margin-top:4px}.top-actions{display:flex;align-items:center;gap:10px}.status-pill{display:flex;align-items:center;gap:8px;padding:8px 11px;border:1px solid var(--line);border-radius:30px;color:var(--muted);font-size:10px}.icon-btn,.menubtn{width:37px;height:37px;border:1px solid var(--line);border-radius:11px;background:var(--panel);color:var(--text);display:grid;place-items:center;cursor:pointer;transition:.2s}.icon-btn:hover{border-color:var(--cyan);color:var(--cyan);transform:translateY(-2px)}.menubtn{display:none}
        .hero{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(330px,.9fr);gap:30px;align-items:center;padding:48px 0 28px;min-height:390px}.eyebrow{display:flex;align-items:center;gap:9px;color:var(--cyan);font-size:9px;font-weight:700;letter-spacing:2px}.eyebrow span{width:19px;height:2px;background:var(--cyan);box-shadow:0 0 10px var(--cyan)}.hero h1{font:800 clamp(36px,4.3vw,58px)/1.07 'Manrope',sans-serif;letter-spacing:-2.4px;margin:19px 0 15px}.hero h1 em{font-style:normal;background:linear-gradient(90deg,#55ead4,#5db9ec);-webkit-background-clip:text;background-clip:text;color:transparent}.hero-copy>p{max-width:550px;color:var(--muted);font-size:14px;line-height:1.8;margin:0}.hero-actions{display:flex;gap:11px;flex-wrap:wrap;margin-top:24px}.btn{display:inline-flex;align-items:center;justify-content:center;gap:9px;border:1px solid transparent;border-radius:10px;padding:11px 16px;font:700 12px 'DM Sans';text-decoration:none;cursor:pointer;transition:transform .2s,box-shadow .2s,border-color .2s,background .2s;color:var(--text)}.btn:hover:not(:disabled){transform:translateY(-2px)}.btn:disabled{opacity:.45;cursor:not-allowed}.btn-primary{background:linear-gradient(110deg,#28cdbd,#238fc0);color:#04232d;box-shadow:0 8px 25px rgba(35,180,183,.19)}.btn-primary:hover:not(:disabled){box-shadow:0 10px 30px rgba(35,210,196,.3)}.btn-ghost{background:rgba(17,55,70,.4);border-color:var(--line)}.btn-ghost:hover{border-color:var(--cyan)}.hero-note{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:10px;margin-top:19px}.hero-note svg{color:var(--cyan)}
        .globe-card{height:300px;position:relative;border:1px solid rgba(69,202,207,.2);border-radius:22px;overflow:hidden;background:radial-gradient(circle at 52% 48%,rgba(20,137,162,.28),transparent 48%),linear-gradient(145deg,#0a2637,#06141f 75%);box-shadow:var(--shadow)}.globe-card:before,.globe-card:after{content:'';position:absolute;border:1px solid rgba(68,225,213,.15);border-radius:50%;width:310px;height:150px;left:50%;top:48%;transform:translate(-50%,-50%) rotate(-22deg)}.globe-card:after{transform:translate(-50%,-50%) rotate(38deg);width:360px;height:190px}.globe{position:absolute;width:215px;height:215px;border-radius:50%;left:50%;top:47%;transform:translate(-50%,-50%);background:radial-gradient(circle at 32% 28%,rgba(79,222,217,.9),rgba(18,112,145,.7) 24%,rgba(6,42,66,.95) 68%,#04131f 100%);box-shadow:inset -25px -18px 45px rgba(0,0,0,.75),0 0 45px rgba(37,196,208,.28),0 0 0 1px rgba(93,232,221,.35);overflow:hidden}.globe:before{content:'';position:absolute;inset:15px;border-radius:50%;background:repeating-linear-gradient(0deg,transparent 0 22px,rgba(98,235,224,.16) 23px 24px),repeating-linear-gradient(90deg,transparent 0 28px,rgba(98,235,224,.13) 29px 30px);opacity:.8}.globe:after{content:'';position:absolute;inset:0;border-radius:50%;background:radial-gradient(ellipse at 35% 35%,rgba(103,241,218,.3),transparent 35%),linear-gradient(110deg,transparent 42%,rgba(0,0,0,.52))}.continent{position:absolute;background:linear-gradient(135deg,#47bfa8,#176e79);filter:drop-shadow(0 0 5px rgba(64,228,204,.3));opacity:.8;z-index:1}.continent.one{width:75px;height:45px;left:40px;top:46px;border-radius:55% 35% 60% 30%;transform:rotate(-25deg)}.continent.two{width:55px;height:82px;right:43px;top:91px;border-radius:45% 55% 40% 60%;transform:rotate(20deg)}.continent.three{width:43px;height:27px;left:91px;bottom:38px;border-radius:60% 35% 55% 40%;transform:rotate(15deg)}.scan-ring{position:absolute;inset:27px;border:1px dashed rgba(75,235,219,.35);border-radius:50%;animation:spin 28s linear infinite}.scan-ring:before{content:'';position:absolute;width:7px;height:7px;border-radius:50%;background:#62f4d9;top:17px;left:27px;box-shadow:0 0 15px #62f4d9}.orbit-line{position:absolute;left:50%;top:50%;width:290px;height:110px;border:1px solid rgba(66,228,208,.25);border-radius:50%;transform:translate(-50%,-50%) rotate(-27deg)}.globe-label{position:absolute;z-index:3;padding:9px 11px;border:1px solid rgba(79,218,207,.25);border-radius:10px;background:rgba(4,25,37,.78);backdrop-filter:blur(10px);font-size:9px;letter-spacing:.6px;color:#bde9e7}.globe-label b{display:block;color:#fff;font-size:10px;margin-top:3px;letter-spacing:0}.globe-label.left{left:15px;top:34px}.globe-label.right{right:14px;bottom:36px}.globe-coord{position:absolute;left:18px;bottom:16px;color:#6e9ca6;font-size:8px;letter-spacing:1px}.globe-badge{position:absolute;right:16px;top:15px;color:#72e8d8;font-size:9px;letter-spacing:1px}
        .metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 42px}.metric-card{display:flex;align-items:center;gap:12px;padding:16px 15px;border:1px solid var(--line);border-radius:14px;background:linear-gradient(140deg,var(--panel),rgba(11,37,51,.56));box-shadow:var(--shadow);transition:.22s}.light .metric-card{background:var(--panel)}.metric-card:hover{transform:translateY(-4px);border-color:rgba(66,228,208,.45)}.metric-icon{width:39px;height:39px;flex:0 0 39px;border-radius:12px;display:grid;place-items:center;color:var(--cyan);background:rgba(47,196,190,.12)}.metric-card b{display:block;font:700 13px 'Manrope';color:var(--text)}.metric-card small{display:block;color:var(--muted);font-size:10px;margin-top:4px}
        .section{padding:20px 0 42px;scroll-margin-top:20px}.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-bottom:19px}.section-head h2{font:800 25px 'Manrope';letter-spacing:-.8px;margin:9px 0 0}.section-sub{color:var(--muted);font-size:11px;max-width:360px;line-height:1.6}.ready{display:flex;align-items:center;gap:8px;color:var(--cyan);font-size:10px;padding:8px 11px;border:1px solid var(--line);border-radius:30px}
        .workspace-grid{display:grid;grid-template-columns:minmax(290px,.78fr) minmax(0,1.5fr);gap:15px}.panel{border:1px solid var(--line);border-radius:17px;background:linear-gradient(145deg,var(--panel),rgba(7,27,39,.82));box-shadow:var(--shadow);padding:19px;min-width:0}.light .panel{background:var(--panel)}.panel-head{display:flex;align-items:center;gap:11px;margin-bottom:17px}.step{width:31px;height:31px;display:grid;place-items:center;border-radius:9px;background:rgba(66,228,208,.12);color:var(--cyan);font:800 10px 'Manrope'}.panel-head h3{font:700 14px 'Manrope';margin:0}.panel-head p{font-size:10px;color:var(--muted);margin:4px 0 0}.panel-badge{margin-left:auto;padding:6px 9px;border-radius:30px;background:rgba(126,150,160,.1);color:var(--muted);font-size:9px}.panel-badge.done{background:rgba(42,207,153,.12);color:#53e2ae}
        .sensor-options{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:13px}.sensor-option{display:flex;align-items:center;gap:8px;text-align:left;padding:10px;border:1px solid var(--line);border-radius:11px;background:rgba(5,24,35,.25);color:var(--text);cursor:pointer;min-width:0}.light .sensor-option{background:#f5fbfc}.sensor-option.selected{border-color:var(--cyan);background:rgba(44,190,184,.1)}.sensor-option>svg{width:17px;height:17px;color:var(--cyan);flex:0 0 auto}.sensor-option b{display:block;font-size:10px}.sensor-option small{display:block;color:var(--muted);font-size:8px;margin-top:3px}.sensor-option .check{margin-left:auto;width:14px}
        .dropzone{min-height:190px;border:1px dashed rgba(76,210,207,.4);border-radius:14px;background:linear-gradient(145deg,rgba(26,128,147,.12),rgba(6,28,41,.16));display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:19px;cursor:pointer;transition:.22s}.light .dropzone{background:#f2fafb}.dropzone:hover,.dropzone.dragging{border-color:var(--cyan);background:rgba(43,199,190,.13);box-shadow:inset 0 0 35px rgba(55,220,206,.08)}.dropzone input{display:none}.upload-icon{width:52px;height:52px;border-radius:16px;display:grid;place-items:center;color:var(--cyan);background:rgba(66,228,208,.12);margin-bottom:12px}.drop-title{font:700 12px 'Manrope';max-width:100%;overflow-wrap:anywhere}.drop-sub{font-size:10px;color:var(--muted);margin-top:6px}.file-types{font-size:8px;font-weight:700;letter-spacing:1.2px;color:var(--cyan);margin-top:14px}.hint{display:flex;gap:9px;padding:11px;margin:12px 0 14px;border-radius:10px;background:rgba(57,132,157,.09);color:var(--muted)}.hint svg{width:15px;flex:0 0 15px;color:var(--cyan)}.hint p{font-size:9px;line-height:1.6;margin:0}.full{width:100%}.privacy{display:flex;align-items:center;justify-content:center;gap:6px;color:var(--muted);font-size:9px;margin-top:12px}.privacy svg{color:var(--cyan)}.error{display:flex;align-items:flex-start;gap:8px;padding:11px;border-radius:10px;background:rgba(238,83,98,.12);border:1px solid rgba(238,83,98,.25);color:#ff9ca7;font-size:10px;line-height:1.5;margin:10px 0}.error svg{flex:0 0 auto}
        .results-panel{min-height:440px}.empty-state{min-height:340px;border:1px dashed var(--line);border-radius:13px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:25px;background:radial-gradient(circle at 50% 35%,rgba(39,154,168,.09),transparent 50%)}.empty-art{width:88px;height:88px;border-radius:25px;display:grid;place-items:center;background:linear-gradient(145deg,rgba(48,200,191,.18),rgba(36,120,168,.12));color:var(--cyan);box-shadow:0 0 40px rgba(48,200,191,.08);margin-bottom:16px}.empty-state h4{font:700 14px 'Manrope';margin:0 0 7px}.empty-state p{max-width:330px;color:var(--muted);font-size:10px;line-height:1.7;margin:0}.legend{display:flex;gap:13px;flex-wrap:wrap;justify-content:center;margin-top:17px}.legend span{display:flex;align-items:center;gap:5px;color:var(--muted);font-size:9px}.legend i,.class-label i{width:7px;height:7px;border-radius:50%;display:inline-block}
        .detected-banner{display:flex;align-items:center;gap:12px;padding:12px 14px;border:1px solid rgba(66,228,208,.2);border-radius:12px;background:linear-gradient(100deg,rgba(37,177,173,.13),rgba(30,108,145,.07));margin-bottom:15px}.detected-icon{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:rgba(66,228,208,.13);color:var(--cyan)}.detected-banner small{display:block;color:var(--muted);font-size:8px;letter-spacing:1.2px}.detected-banner h4{font:700 13px 'Manrope';margin:4px 0 0}.detected-banner p{font-size:9px;color:var(--muted);margin:3px 0 0}.image-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.image-card{min-width:0}.image-card h5{font:700 10px 'Manrope';margin:0 0 8px}.image-view{position:relative;width:100%;aspect-ratio:1/1;border:1px solid var(--line);border-radius:11px;overflow:hidden;background:#061923;display:block;padding:0;cursor:zoom-in}.image-view img{width:100%;height:100%;object-fit:contain;display:block;transition:transform .3s}.image-view:hover img{transform:scale(1.045)}.image-placeholder{aspect-ratio:1/1;border:1px dashed var(--line);border-radius:11px;display:grid;place-items:center;padding:10px;text-align:center;color:var(--muted);font-size:9px}.zoom-hint{font-size:8px;color:var(--muted);display:block;margin-top:6px}.stats-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin:15px 0}.stat-box{padding:11px;border:1px solid var(--line);border-radius:10px;background:rgba(6,26,37,.28)}.light .stat-box{background:#f5fbfc}.stat-box small{display:block;color:var(--muted);font-size:8px}.stat-box b{display:block;font:700 16px 'Manrope';margin-top:5px;color:var(--cyan)}.breakdown{padding:13px;border:1px solid var(--line);border-radius:12px;background:rgba(4,22,32,.18)}.breakdown-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}.breakdown h5{font:700 11px 'Manrope';margin:0}.breakdown-head span{font-size:8px;color:var(--muted)}.class-row{margin:10px 0}.class-row-top{display:flex;justify-content:space-between;gap:10px;font-size:9px;margin-bottom:6px}.class-label{display:flex;align-items:center;gap:7px;color:var(--muted)}.class-row-top b{font-size:9px}.track{height:5px;border-radius:20px;background:rgba(130,163,174,.14);overflow:hidden}.track span{height:100%;display:block;border-radius:20px;transition:width .6s ease}.result-actions{display:flex;gap:9px;flex-wrap:wrap;margin-top:14px}.result-actions .btn{flex:1}.caveat,.result-message{font-size:9px;line-height:1.6;color:var(--muted);margin:12px 0 0}
        .analytics-grid{display:grid;grid-template-columns:1.1fr .9fr 1fr;gap:13px}.analytics-card{border:1px solid var(--line);border-radius:15px;padding:17px;background:var(--panel);min-height:180px}.analytics-card h3{font:700 12px 'Manrope';margin:0 0 6px}.analytics-card>p{font-size:9px;color:var(--muted);margin:0}.donut-wrap{display:flex;align-items:center;gap:18px;margin-top:18px}.donut{width:112px;height:112px;flex:0 0 112px;border-radius:50%;background:conic-gradient(#27d8c1 0 72%,#f3a64b 72% 82%,#df6b9b 82% 89%,#548fc9 89% 95%,#9d8bce 95% 100%);position:relative;display:grid;place-items:center}.donut:after{content:'Classes';position:absolute;inset:19px;border-radius:50%;background:var(--bg2);display:grid;place-items:center;font-size:10px;color:var(--muted)}.donut-legend{display:grid;gap:8px}.donut-legend span{display:flex;align-items:center;gap:7px;font-size:9px;color:var(--muted)}.donut-legend i{width:7px;height:7px;border-radius:50%;background:var(--cyan)}.severity-meter{height:9px;border-radius:20px;background:linear-gradient(90deg,#32c99c,#e7c44e,#ed795c);margin:25px 0 10px;position:relative}.severity-pin{position:absolute;top:-5px;width:18px;height:18px;border:3px solid var(--panel);border-radius:50%;background:var(--cyan);box-shadow:0 0 0 1px var(--cyan);left:50%;transform:translateX(-50%)}.severity-scale{display:flex;justify-content:space-between;font-size:8px;color:var(--muted)}.severity-value{font:800 22px 'Manrope';margin-top:18px}.severity-value span{display:block;color:var(--muted);font:400 9px 'DM Sans';margin-top:4px}.area-number{font:800 27px 'Manrope';color:var(--cyan);margin-top:21px}.area-number small{font:500 10px 'DM Sans';color:var(--muted)}.report-card{display:flex;align-items:center;gap:12px;margin-top:17px}.report-icon{width:42px;height:42px;border-radius:12px;display:grid;place-items:center;background:rgba(66,228,208,.12);color:var(--cyan)}.report-card p{font-size:9px;line-height:1.5;color:var(--muted);margin:0}
        .workflow-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.workflow-card{padding:17px;border:1px solid var(--line);border-radius:14px;background:var(--panel);transition:.22s}.workflow-card:hover{transform:translateY(-4px);border-color:rgba(66,228,208,.4)}.workflow-card small{color:var(--cyan);font-size:8px;font-weight:700;letter-spacing:1.3px}.workflow-icon{width:38px;height:38px;border-radius:11px;background:rgba(66,228,208,.12);color:var(--cyan);display:grid;place-items:center;margin:14px 0}.workflow-card h3{font:700 12px 'Manrope';margin:0 0 7px}.workflow-card p{font-size:9px;line-height:1.6;color:var(--muted);margin:0}
        .notice{display:flex;gap:13px;padding:17px 19px;border:1px solid rgba(235,178,85,.2);border-radius:14px;background:rgba(171,125,44,.07);margin:10px 0 25px}.notice>svg{color:#e9b85f;flex:0 0 auto}.notice b{font:700 11px 'Manrope'}.notice p{font-size:9px;line-height:1.65;color:var(--muted);margin:5px 0 0}.footer{display:flex;align-items:center;justify-content:space-between;gap:15px;padding:20px 0;border-top:1px solid var(--line);color:var(--muted);font-size:9px}.footer-brand{display:flex;align-items:center;gap:8px;color:var(--text);font:800 13px 'Manrope'}.footer-brand svg{color:var(--cyan)}
        .spin{animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}@keyframes pulse{0%,100%{opacity:.65}50%{opacity:1}}
        .modal-backdrop{position:fixed;inset:0;z-index:100;background:rgba(1,10,16,.91);backdrop-filter:blur(12px);display:flex;align-items:center;justify-content:center;padding:20px}.modal{width:min(100%,1050px);height:min(88vh,850px);display:flex;flex-direction:column;gap:12px;color:white}.modal-head{display:flex;align-items:center;gap:9px}.modal-head strong{margin-right:auto;font:700 14px 'Manrope'}.modal-view{flex:1;min-height:0;overflow:auto;display:flex;align-items:center;justify-content:center;border:1px solid rgba(140,190,200,.25);border-radius:13px;background:#081720}.modal-view img{max-width:none;object-fit:contain}.modal-foot{font-size:9px;color:#9bb4bd}
        @media(max-width:1100px){.app-layout{grid-template-columns:210px minmax(0,1fr)}.main-area{padding:0 22px 25px}.hero{gap:18px}.metric-grid{grid-template-columns:repeat(2,1fr)}.analytics-grid{grid-template-columns:1fr 1fr}.analytics-card:last-child{grid-column:span 2}}
        @media(max-width:760px){.app-layout{display:block}.sidebar{position:fixed;left:0;top:0;bottom:0;width:260px;height:100dvh;transform:translateX(-105%);transition:transform .25s;box-shadow:var(--shadow)}.sidebar.mobile-open{transform:translateX(0)}.main-area{padding:0 16px 22px}.topbar{height:65px}.menubtn{display:grid}.status-pill{display:none}.hero{grid-template-columns:1fr;padding:35px 0 22px;min-height:0}.hero h1{font-size:42px}.globe-card{height:260px}.metric-grid{gap:8px;margin-bottom:25px}.metric-card{padding:12px 10px;gap:8px}.metric-icon{width:33px;height:33px;flex-basis:33px}.metric-card b{font-size:11px}.metric-card small{font-size:8px}.workspace-grid{grid-template-columns:1fr}.section-head{align-items:flex-start;flex-direction:column}.section-sub{max-width:none}.analytics-grid{grid-template-columns:1fr}.analytics-card:last-child{grid-column:auto}.workflow-grid{grid-template-columns:1fr 1fr}.footer{flex-wrap:wrap}.results-panel{min-height:0}}
        @media(max-width:450px){.hero h1{font-size:35px}.metric-grid{grid-template-columns:1fr 1fr}.metric-card{align-items:flex-start}.metric-card small{line-height:1.35}.image-grid{gap:6px}.image-card h5{font-size:8px}.stats-grid{gap:6px}.stat-box{padding:9px}.stat-box b{font-size:13px}.sensor-options{grid-template-columns:1fr}.workflow-grid{grid-template-columns:1fr}.panel{padding:14px}.result-actions{flex-direction:column}.result-actions .btn{width:100%}.donut-wrap{gap:12px}}
        @media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.01ms!important;transition-duration:.01ms!important}}
      `}</style>

      <div className="app-layout">
        <aside className={`sidebar ${menu ? "mobile-open" : ""}`}>
          <a href="#dashboard" className="brand" onClick={() => setMenu(false)}>
            <span className="brandmark"><Waves size={23} /></span>
            <span className="brandtext">oceanmind<span>ai</span><small>MARINE INTELLIGENCE</small></span>
          </a>
          <div className="side-label">WORKSPACE</div>
          <nav className="side-nav">
            <a className={activeSection === "dashboard" ? "active" : ""} href="#dashboard" onClick={() => { setActiveSection("dashboard"); setMenu(false); }}><Globe2 /> Dashboard</a>
            <a className={activeSection === "workspace" ? "active" : ""} href="#workspace" onClick={() => { setActiveSection("workspace"); setMenu(false); }}><Satellite /> Analyze image</a>
            <a className={activeSection === "results" ? "active" : ""} href="#results" onClick={() => { setActiveSection("results"); setMenu(false); }}><Layers3 /> Results</a>
            <a className={activeSection === "analytics" ? "active" : ""} href="#analytics" onClick={() => { setActiveSection("analytics"); setMenu(false); }}><BarChart3 /> Analytics</a>
            <a className={activeSection === "reports" ? "active" : ""} href="#reports" onClick={() => { setActiveSection("reports"); setMenu(false); }}><FileText /> Reports</a>
          </nav>
          <div className="side-label">INFORMATION</div>
          <nav className="side-nav">
            <a className={activeSection === "workflow" ? "active" : ""} href="#workflow" onClick={() => { setActiveSection("workflow"); setMenu(false); }}><Activity /> How it works</a>
            <a className={activeSection === "about" ? "active" : ""} href="#about" onClick={() => { setActiveSection("about"); setMenu(false); }}><Info /> About project</a>
          </nav>
          <div className="sidebar-bottom">
            <b>Cleaner oceans start with insight.</b>
            <p>Explore satellite imagery with AI-assisted marine pollution segmentation.</p>
            <div className="side-status"><span className="dot" /> Research workspace</div>
          </div>
        </aside>

        <div className="main-area">
          <header className="topbar">
            <div className="top-title">OceanMind AI <small>AI-powered marine pollution detection & segmentation</small></div>
            <div className="top-actions">
              <span className="status-pill"><span className="dot" /> Workspace ready</span>
              <button className="icon-btn" onClick={() => setLightMode(!lightMode)} aria-label="Toggle theme" title="Toggle light and dark theme">
                {lightMode ? <Moon size={16} /> : <Sun size={16} />}
              </button>
              <button className="menubtn" onClick={() => setMenu(!menu)} aria-label="Toggle navigation">{menu ? <X /> : <Menu />}</button>
            </div>
          </header>

          <section id="dashboard" className="hero">
            <div className="hero-copy">
              <div className="eyebrow"><span /> OCEAN HEALTH MONITORING PLATFORM</div>
              <h1>Detect. Analyze.<br /><em>Protect our oceans.</em></h1>
              <p>Turn satellite imagery into actionable marine intelligence with AI-assisted segmentation, pollution mapping, and visual reporting.</p>
              <div className="hero-actions">
                <a href="#workspace" className="btn btn-primary" onClick={() => setActiveSection("workspace")}>Start an analysis <ArrowRight size={16} /></a>
                <a href="#workflow" className="btn btn-ghost" onClick={() => setActiveSection("workflow")}>Explore the workflow <ChevronRight size={15} /></a>
              </div>
              <div className="hero-note"><ShieldCheck size={15} /> Research prototype · Predictions require validation</div>
            </div>
            <div className="globe-card" aria-label="Decorative satellite ocean visualization">
              <div className="globe-badge"><Sparkles size={11} style={{ display: "inline", marginRight: 5 }} /> AI OCEAN INTELLIGENCE</div>
              <div className="orbit-line" />
              <div className="globe"><div className="continent one" /><div className="continent two" /><div className="continent three" /><div className="scan-ring" /></div>
              <div className="globe-label left">SATELLITE SCAN<b>Ocean surface mapping</b></div>
              <div className="globe-label right">AI SEGMENTATION<b>Pixel-level predictions</b></div>
              <div className="globe-coord">EARTH OBSERVATION · MARINE MONITORING</div>
            </div>
          </section>

          <section className="metric-grid" aria-label="Platform capabilities">
            <div className="metric-card"><span className="metric-icon"><Layers3 size={19} /></span><div><b>15 Classes</b><small>MARIDA segmentation</small></div></div>
            <div className="metric-card"><span className="metric-icon"><Microscope size={19} /></span><div><b>U-Net Model</b><small>Deep learning model</small></div></div>
            <div className="metric-card"><span className="metric-icon"><Satellite size={19} /></span><div><b>Satellite imagery</b><small>11-band optical input</small></div></div>
            <div className="metric-card"><span className="metric-icon"><Gauge size={19} /></span><div><b>Pixel-level</b><small>Segmentation output</small></div></div>
          </section>

          <section id="workspace" className="section">
            <div className="section-head">
              <div><div className="eyebrow"><span /> ANALYSIS WORKSPACE</div><h2>Turn imagery into insight</h2></div>
              <div className="ready"><span className="dot" /> Ready for upload</div>
            </div>
            <div className="workspace-grid">
              <div className="panel">
                <div className="panel-head"><b className="step">01</b><div><h3>Input imagery</h3><p>Select the sensor and upload your satellite image.</p></div></div>
                <div className="sensor-options">
                  <button className={`sensor-option ${sensor === "optical" ? "selected" : ""}`} onClick={() => { setSensor("optical"); setResult(null); setError(""); }}><Globe2 /><span><b>Optical</b><small>Sentinel-2 / multispectral</small></span>{sensor === "optical" && <CheckCircle2 className="check" />}</button>
                  <button className={`sensor-option ${sensor === "sar" ? "selected" : ""}`} onClick={() => { setSensor("sar"); setResult(null); setError(""); }}><Waves /><span><b>Radar (SAR)</b><small>Not supported by checkpoint</small></span>{sensor === "sar" && <CheckCircle2 className="check" />}</button>
                </div>
                <label className={`dropzone ${dragging ? "dragging" : ""}`} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={drop}>
                  <input type="file" accept=".tif,.tiff,.png,.jpg,.jpeg,image/*" onChange={fileChange} />
                  <span className="upload-icon">{file ? <FileImage size={27} /> : <CloudUpload size={29} />}</span>
                  <b className="drop-title">{file ? file.name : "Drop your satellite image here"}</b>
                  <span className="drop-sub">{file ? `${(file.size / 1048576).toFixed(2)} MB · Selected` : "or click to browse files"}</span>
                  <span className="file-types">{file ? "CLICK TO CHOOSE ANOTHER FILE" : "11-BAND GEOTIFF · TIFF"}</span>
                </label>
                <div className="hint"><Info size={15} /><p>For this trained model, upload the original 11-band, 256 × 256 optical GeoTIFF. Label masks ending in <code>_cl.tif</code> and confidence masks ending in <code>_conf.tif</code> are not model inputs.</p></div>
                {error && <div className="error" role="alert"><X size={15} />{error}</div>}
                <button className="btn btn-primary full" onClick={analyze} disabled={!file || busy}>
                  {busy ? <><LoaderCircle className="spin" size={16} /> Analyzing image...</> : <>Run pollution analysis <ArrowRight size={16} /></>}
                </button>
                <div className="privacy"><ShieldCheck size={13} /> Uploaded imagery is processed by the configured API.</div>
              </div>

              <div id="results" className="panel results-panel">
                <div className="panel-head"><b className="step">02</b><div><h3>Analysis results</h3><p>Review predictions returned by the U-Net model.</p></div><span className={`panel-badge ${result ? "done" : ""}`}>{result ? "Complete" : busy ? "Processing" : "Awaiting input"}</span></div>
                {!result ? (
                  <div className="empty-state">
                    <div className="empty-art">{busy ? <LoaderCircle className="spin" size={33} /> : <Microscope size={32} />}</div>
                    <h4>{busy ? "Your image is being analyzed" : "Your analysis will appear here"}</h4>
                    <p>{busy ? "The U-Net model is processing the uploaded satellite imagery. This may take a little while." : "Upload a valid satellite GeoTIFF and run analysis to view the predicted mask, overlay, and class statistics."}</p>
                    <div className="legend"><span><i style={{ background: colors["Marine Debris"] }} />Marine debris</span><span><i style={{ background: colors["Dense Sargassum"] }} />Sargassum</span><span><i style={{ background: colors["Marine Water"] }} />Marine water</span></div>
                  </div>
                ) : (
                  <div className="result-content">
                    <div className="detected-banner"><span className="detected-icon"><Droplets size={21} /></span><div><small>MODEL PREDICTION</small><h4>{detected}</h4><p>{result.severity ? `Severity: ${result.severity}` : "Predicted class"}</p></div></div>
                    <div className="image-grid">
                      {imagePanels.map((item) => <div className="image-card" key={item.title}>
                        <h5>{item.title}</h5>
                        {item.url ? <button type="button" className="image-view" onClick={() => { setZoomImage({ title: item.title, url: imageUrl(item.url) }); setZoomLevel(1); }} title="Click to zoom and inspect"><img src={imageUrl(item.url)} alt={item.alt} /></button> : <div className="image-placeholder">{item.title === "Original image" ? "Original preview not returned by API" : "Image unavailable from API"}</div>}
                        {item.url && <small className="zoom-hint"><ZoomIn size={10} style={{ display: "inline", verticalAlign: "middle" }} /> Click to zoom</small>}
                      </div>)}
                    </div>
                    <div className="stats-grid">
                      <div className="stat-box"><small>Predicted debris area</small><b>{area}</b></div>
                      <div className="stat-box"><small>Mean max probability</small><b>{confidence}</b></div>
                      <div className="stat-box"><small>Dice score</small><b>{typeof result.dice === "number" ? result.dice.toFixed(3) : "—"}</b></div>
                    </div>
                    {result.message && <p className="result-message">{result.message}</p>}
                    {classRows.length > 0 && <div className="breakdown">
                      <div className="breakdown-head"><h5>Predicted class breakdown</h5><span>{classRows.length} classes returned</span></div>
                      {classRows.map(([name, value]) => <div className="class-row" key={name}>
                        <div className="class-row-top"><span className="class-label"><i style={{ background: colors[name] || "#9aa8b6" }} />{name}</span><b>{value.toFixed(2)}%</b></div>
                        <div className="track"><span style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: colors[name] || "#9aa8b6" }} /></div>
                      </div>)}
                    </div>}
                    <div className="result-actions">
                      <button className="btn btn-primary" disabled={!overlayUrl} onClick={() => download(overlayUrl, "oceanmind-overlay.png")}><ArrowDownToLine size={15} /> Download overlay</button>
                      <button className="btn btn-ghost" disabled={!maskUrl} onClick={() => download(maskUrl, "oceanmind-mask.png")}><ArrowDownToLine size={15} /> Download mask</button>
                    </div>
                    <p className="caveat">Experimental predictions require validation. Mean maximum probability is not validated accuracy. A Dice score is meaningful only when ground-truth labels are supplied for evaluation.</p>
                  </div>
                )}
              </div>
            </div>
          </section>

          <section id="analytics" className="section">
            <div className="section-head"><div><div className="eyebrow"><span /> INSIGHTS & METRICS</div><h2>Pollution analytics</h2></div><p className="section-sub">Live result metrics appear here after a successful analysis. Decorative indicators are illustrative until then.</p></div>
            <div className="analytics-grid">
              <div className="analytics-card"><h3>Class distribution</h3><p>{result ? "Distribution from the current prediction" : "Illustrative class distribution"}</p><div className="donut-wrap"><div className="donut" /><div className="donut-legend"><span><i style={{ background: "#27d8c1" }} />Marine water</span><span><i style={{ background: "#f3a64b" }} />Sargassum</span><span><i style={{ background: "#df6b9b" }} />Debris / ships</span><span><i style={{ background: "#548fc9" }} />Other classes</span></div></div></div>
              <div className="analytics-card"><h3>Pollution severity</h3><p>{result ? "Model-reported severity" : "Awaiting prediction"}</p><div className="severity-value">{result?.severity || "—"}<span>{result ? "Reported by the model" : "Run an analysis to populate this metric"}</span></div><div className="severity-meter"><span className="severity-pin" /></div><div className="severity-scale"><span>Low</span><span>Moderate</span><span>High</span></div></div>
              <div className="analytics-card"><h3>Affected-area indicator</h3><p>Predicted debris area percentage</p><div className="area-number">{result ? area : "—"} <small>{result ? "of analyzed pixels" : "awaiting result"}</small></div><div className="report-card"><span className="report-icon"><Activity size={20} /></span><p>{result ? "A prediction is available. Review the segmentation mask and overlay before interpreting the result." : "Run an image analysis to view the model's area estimate and output summary."}</p></div></div>
            </div>
          </section>

          <section id="reports" className="section">
            <div className="section-head"><div><div className="eyebrow"><span /> EXPORT & DOCUMENTATION</div><h2>Analysis report</h2></div></div>
            <div className="panel" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 18, flexWrap: "wrap" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 13 }}><span className="report-icon"><FileText size={22} /></span><div><h3 style={{ font: "700 13px Manrope", margin: "0 0 5px" }}>Export your analysis summary</h3><p style={{ color: "var(--muted)", fontSize: 10, margin: 0 }}>Generate a report containing the prediction, metrics and class breakdown.</p></div></div>
              <button className="btn btn-primary" disabled={!result} onClick={downloadReport}><ArrowDownToLine size={15} /> Generate & download report</button>
            </div>
          </section>

          <section id="workflow" className="section">
            <div className="section-head"><div><div className="eyebrow"><span /> FROM IMAGERY TO INSIGHT</div><h2>How OceanMind AI works</h2></div><p className="section-sub">A clear, traceable workflow from satellite input to predicted map.</p></div>
            <div className="workflow-grid">
              <article className="workflow-card"><small>01 / INPUT</small><span className="workflow-icon"><CloudUpload size={19} /></span><h3>Upload imagery</h3><p>Provide a supported 11-band optical satellite GeoTIFF.</p></article>
              <article className="workflow-card"><small>02 / MODEL</small><span className="workflow-icon"><Layers3 size={19} /></span><h3>Segment with U-Net</h3><p>The trained model predicts one of 15 MARIDA classes per pixel.</p></article>
              <article className="workflow-card"><small>03 / REVIEW</small><span className="workflow-icon"><MapPinned size={19} /></span><h3>Inspect the map</h3><p>Review predicted classes and generated mask and overlay.</p></article>
              <article className="workflow-card"><small>04 / EXPORT</small><span className="workflow-icon"><ArrowDownToLine size={19} /></span><h3>Export results</h3><p>Download the generated mask, overlay and report.</p></article>
            </div>
          </section>

          <section id="about" className="notice"><ShieldCheck size={20} /><div><b>Research prototype — not an operational alert system</b><p>Clouds, waves, satellite artefacts and other water types can be misclassified. Predictions require validation against labeled data and expert review before environmental decisions. SAR input and real-world area estimates are not supported by this checkpoint.</p></div></section>

          <footer className="footer"><span className="footer-brand"><Waves size={18} /> OceanMind AI</span><span>Built for healthier oceans · OceanMind AI 2.0</span><span className="side-status"><span className="dot" /> Frontend workspace</span></footer>
        </div>
      </div>

      {zoomImage && <div className="modal-backdrop" onClick={() => setZoomImage(null)}>
        <div className="modal" onClick={(e) => e.stopPropagation()}>
          <div className="modal-head"><strong>{zoomImage.title}</strong>
            <button className="icon-btn" onClick={() => setZoomLevel(v => Math.max(1, v - .5))} aria-label="Zoom out">−</button>
            <span style={{ fontSize: 10 }}>{Math.round(zoomLevel * 100)}%</span>
            <button className="icon-btn" onClick={() => setZoomLevel(v => Math.min(5, v + .5))} aria-label="Zoom in">+</button>
            <button className="icon-btn" onClick={() => setZoomLevel(1)} aria-label="Reset zoom"><RefreshCw size={14} /></button>
            <button className="icon-btn" onClick={() => setZoomImage(null)} aria-label="Close image"><X size={15} /></button>
          </div>
          <div className="modal-view"><img src={zoomImage.url} alt={zoomImage.title} style={{ width: `${zoomLevel * 100}%`, maxHeight: zoomLevel === 1 ? "100%" : "none" }} /></div>
          <div className="modal-foot">Use + and − to inspect predicted regions. Click outside the image to close.</div>
        </div>
      </div>}
    </main>
  );
}
