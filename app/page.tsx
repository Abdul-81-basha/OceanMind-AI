
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

  async function analyze() {
    if (!file) {
      setError("Upload a satellite image first.");
      return;
    }

    if (sensor === "sar") {
      setError(
        "The current U-Net was trained on 11-band optical imagery. SAR analysis is not supported by this checkpoint."
      );
      return;
    }

    const name = file.name.toLowerCase();
    if (!name.endsWith(".tif") && !name.endsWith(".tiff")) {
      setError(
        "The current model requires an 11-band, 256 × 256 optical GeoTIFF. PNG and JPEG files cannot be used directly."
      );
      return;
    }

    setBusy(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("file", file);
    form.append("sensor_type", sensor);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: form
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          data.detail || data.message || `Analysis failed (${response.status}).`
        );
      }

      setResult(data as Result);
    } catch (e) {
      const msg =
        e instanceof Error ? e.message : "Unable to connect to API.";

      setError(
        msg.includes("Failed to fetch")
          ? `Cannot reach backend at ${API_URL}. Make sure FastAPI is running.`
          : msg
      );
    } finally {
      setBusy(false);
    }
  }

  async function download(url?: string | null, name = "oceanmind-result.png") {
    if (!url) {
      setError("The requested output is not available to download.");
      return;
    }

    try {
      const response = await fetch(imageUrl(url));
      if (!response.ok) {
        throw new Error(`Download failed with HTTP ${response.status}. Run a new analysis and try again.`);
      }

      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
    } catch (err) {
      console.error("OceanMind download error:", err);
      setError(
        err instanceof Error
          ? `${err.message} If the error mentions CORS, check the backend CORS settings for https://ocean-mind-ai.vercel.app.`
          : "Could not download the output. Please run the analysis again and retry."
      );
    }
  }

  function downloadReport() {
    if (!result) {
      setError("Run an analysis before generating a report.");
      return;
    }

    const escapeHtml = (value: unknown) =>
      String(value ?? "—")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");

    const confidenceValue =
      typeof result.confidence === "number"
        ? result.confidence <= 1
          ? result.confidence * 100
          : result.confidence
        : null;
    const classRows = Object.entries(result.class_percentages || {})
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => `
        <tr><td>${escapeHtml(name)}</td><td>${Number(value).toFixed(2)}%</td></tr>
      `)
      .join("");

    const outputLink = (label: string, path?: string | null) => {
      if (!path) return `<li>${escapeHtml(label)}: Not available</li>`;
      const href = imageUrl(path);
      return `<li><a href="${escapeHtml(href)}">${escapeHtml(label)}</a></li>`;
    };

    const reportHtml = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>OceanMind AI 2.0 - Analysis Report</title>
<style>
  :root { color-scheme: light; }
  body { font-family: Arial, Helvetica, sans-serif; color: #173b3a; margin: 0; background: #f3f8f7; }
  main { max-width: 900px; margin: 32px auto; padding: 36px; background: white; border-radius: 16px; box-shadow: 0 8px 28px #123b3a14; }
  header { border-bottom: 3px solid #24c7b5; padding-bottom: 20px; margin-bottom: 24px; }
  h1 { margin: 0 0 8px; font-size: 30px; } h2 { margin-top: 28px; font-size: 18px; }
  .sub { color: #58716f; } .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
  .metric { background: #f1f8f6; border: 1px solid #dbeae6; padding: 16px; border-radius: 10px; }
  .metric small { display: block; color: #58716f; margin-bottom: 8px; } .metric strong { font-size: 20px; }
  table { border-collapse: collapse; width: 100%; margin-top: 12px; } th, td { text-align: left; border-bottom: 1px solid #e1eae8; padding: 10px; } th { background: #f1f8f6; }
  li { margin: 8px 0; overflow-wrap: anywhere; } a { color: #087f75; }
  .note { padding: 14px; background: #fff8e8; border-left: 4px solid #e5b65c; border-radius: 6px; line-height: 1.5; }
  .toolbar { display: flex; gap: 10px; margin-bottom: 20px; } button { border: 0; background: #087f75; color: white; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: bold; }
  @media (max-width: 600px) { main { margin: 0; padding: 20px; border-radius: 0; } .grid { grid-template-columns: 1fr; } }
  @media print { body { background: white; } main { box-shadow: none; margin: 0; max-width: none; padding: 0; } .toolbar { display: none; } a { color: #087f75; text-decoration: none; } }
</style>
</head>
<body>
<main>
  <div class="toolbar"><button onclick="window.print()">Print / Save as PDF</button></div>
  <header>
    <h1>OceanMind AI 2.0</h1>
    <div class="sub">Marine Pollution Segmentation — Analysis Report</div>
    <p class="sub">Generated: ${escapeHtml(new Date().toLocaleString())}</p>
  </header>
  <h2>Prediction summary</h2>
  <div class="grid">
    <div class="metric"><small>Predicted result</small><strong>${escapeHtml(result.pollution_type || result.detected_class || "Analysis complete")}</strong></div>
    <div class="metric"><small>Detected class</small><strong>${escapeHtml(result.detected_class || "Not provided")}</strong></div>
    <div class="metric"><small>Severity</small><strong>${escapeHtml(result.severity || "Not provided")}</strong></div>
    <div class="metric"><small>Predicted marine debris area</small><strong>${typeof result.pollution_area_percent === "number" ? `${result.pollution_area_percent.toFixed(3)}%` : "Not provided"}</strong></div>
    <div class="metric"><small>Mean maximum probability (not validated accuracy)</small><strong>${confidenceValue === null ? "Not provided" : `${confidenceValue.toFixed(2)}%`}</strong></div>
    <div class="metric"><small>Sensor type</small><strong>${escapeHtml(result.sensor_type || sensor)}</strong></div>
  </div>
  <h2>Class breakdown</h2>
  ${classRows ? `<table><thead><tr><th>Predicted class</th><th>Pixel percentage</th></tr></thead><tbody>${classRows}</tbody></table>` : "<p>No class breakdown was returned by the API.</p>"}
  <h2>Generated output files</h2>
  <ul>
    ${outputLink("Original image preview", result.processed_image_url)}
    ${outputLink("Segmentation mask", result.mask_url)}
    ${outputLink("Prediction overlay", result.overlay_url || result.output_image_url)}
  </ul>
  <h2>Model note</h2>
  <p class="note">${escapeHtml(result.message || "Experimental model prediction only. Results require independent validation before environmental use.")} These predictions are experimental and should not be used as the sole basis for environmental decisions.</p>
  <p class="sub">Report generated by OceanMind AI 2.0. Predictions require independent validation against labeled data and expert review.</p>
</main>
</body>
</html>`;

    const blob = new Blob([reportHtml], { type: "text/html;charset=utf-8" });
    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = blobUrl;
    link.download = "oceanmind-analysis-report.html";
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(blobUrl), 10000);
  }

  const detected =
    result?.pollution_type || result?.detected_class || "Analysis complete";

  const area =
    typeof result?.pollution_area_percent === "number"
      ? `${result.pollution_area_percent.toFixed(3)}%`
      : "—";

  const confidence =
    typeof result?.confidence === "number"
      ? `${(
          result.confidence <= 1
            ? result.confidence * 100
            : result.confidence
        ).toFixed(1)}%`
      : "Not provided";

  const maskUrl = result?.mask_url;
  const overlayUrl = result?.overlay_url || result?.output_image_url;
  const originalUrl = previewUrl || result?.processed_image_url;

  const imagePanels = [
    {
      title: "Original image",
      url: originalUrl,
      alt: "Original satellite image"
    },
    {
      title: "Segmentation mask",
      url: maskUrl,
      alt: "Predicted segmentation mask"
    },
    {
      title: "Prediction overlay",
      url: overlayUrl,
      alt: "Color-coded prediction overlay"
    }
  ];

  return (
    <main className="shell">
      <header className="topbar">
        <a href="#" className="brand">
          <span className="brandmark"><Waves size={23} /></span>
          <span className="brandtext">
            oceanmind<span>ai</span>
            <small>MARINE INTELLIGENCE</small>
          </span>
        </a>

        <nav className={menu ? "nav open" : "nav"}>
          <a href="#workspace" onClick={() => setMenu(false)}>Workspace</a>
          <a href="#workflow" onClick={() => setMenu(false)}>How it works</a>
          <a href="#about" onClick={() => setMenu(false)}>About</a>
        </nav>

        <div className="topright">
          <span className="status"><i /> AI workspace</span>
          <button
            className="menubtn"
            onClick={() => setMenu(!menu)}
            aria-label="Toggle navigation"
          >
            {menu ? <X /> : <Menu />}
          </button>
        </div>
      </header>

      <section className="hero">
        <div className="herocopy">
          <div className="eyebrow"><span /> OCEAN HEALTH MONITORING PLATFORM</div>
          <h1>See the pollution.<br /><em>Protect the ocean.</em></h1>
          <p>
            Turn satellite imagery into actionable marine intelligence
            with AI-assisted segmentation, pollution mapping, and visual reporting.
          </p>
          <div className="actions">
            <a href="#workspace" className="btn primary">
              Start an analysis <ArrowRight size={17} />
            </a>
            <a href="#workflow" className="btn quiet">Explore the workflow</a>
          </div>
          <div className="trust">
            <ShieldCheck size={16} />
            Research prototype · Results require validation
          </div>
        </div>

        <div className="visual">
          <div className="orbit o1" />
          <div className="orbit o2" />
          <div className="ocean">
            <div className="land l1" />
            <div className="land l2" />
            <div className="gridlines" />
            <span className="hot h1" />
            <span className="hot h2" />
            <span className="hot h3" />
            <div className="maplabel ml1"><i /> DEBRIS CLUSTER<b>Detected region</b></div>
            <div className="maplabel ml2"><i /> REVIEW AREA<b>Needs validation</b></div>
            <span className="coords c1">14Â° 35&apos; N / 72Â° 18&apos; E</span>
            <span className="coords c2">SATELLITE IMAGERY · CONCEPT</span>
          </div>
          <div className="floatcard">
            <span className="floaticon"><Activity size={17} /></span>
            <span><b>Pixel-level mapping</b><small>Segmentation-ready workflow</small></span>
            <strong>AI</strong>
          </div>
          <div className="sidenote"><Globe2 size={14} /> MULTI-SENSOR ANALYSIS</div>
        </div>
      </section>

      <section className="capabilities">
        <div><span><Layers3 /></span><p><b>Multi-class</b><small>Pixel segmentation</small></p></div>
        <div><span><Globe2 /></span><p><b>Optical satellite data</b><small>11-band model input</small></p></div>
        <div><span><Gauge /></span><p><b>Measured metrics</b><small>Area and available scores</small></p></div>
        <div><span><ShieldCheck /></span><p><b>Transparent results</b><small>Experimental predictions</small></p></div>
      </section>

      <section id="workspace" className="section">
        <div className="sectionhead">
          <div>
            <div className="eyebrow"><span /> ANALYSIS WORKSPACE</div>
            <h2>Analyze a satellite image</h2>
          </div>
          <div className="ready"><i /> Ready for upload</div>
        </div>

        <div className="workspace">
          <div className="panel">
            <div className="panelhead">
              <b className="step">01</b>
              <div><h3>Input imagery</h3><p>Select the sensor that matches your file.</p></div>
            </div>

            <div className="sensors">
              <button
                className={sensor === "optical" ? "sensor selected" : "sensor"}
                onClick={() => { setSensor("optical"); setResult(null); setError(""); }}
              >
                <span className="sensoricon optical"><Globe2 /></span>
                <span><b>Optical</b><small>Sentinel-2 / multispectral</small></span>
                {sensor === "optical" && <CheckCircle2 className="check" />}
              </button>
              <button
                className={sensor === "sar" ? "sensor selected" : "sensor"}
                onClick={() => { setSensor("sar"); setResult(null); setError(""); }}
              >
                <span className="sensoricon radar"><Waves /></span>
                <span><b>Radar (SAR)</b><small>Not supported by current model</small></span>
                {sensor === "sar" && <CheckCircle2 className="check" />}
              </button>
            </div>

            <label
              className={`dropzone ${dragging ? "dragging" : ""} ${file ? "hasfile" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={drop}
            >
              <input
                type="file"
                accept=".tif,.tiff,.png,.jpg,.jpeg,image/*"
                onChange={fileChange}
              />
              <span className="uploadicon">
                {file ? <FileImage size={28} /> : <CloudUpload size={30} />}
              </span>
              <b className="droptitle">{file ? file.name : "Drop your satellite image here"}</b>
              <span className="dropsub">
                {file ? `${(file.size / 1048576).toFixed(2)} MB · Selected` : "or click to browse files"}
              </span>
              <span className="filetypes">{file ? "CHOOSE ANOTHER FILE" : "11-BAND GEOTIFF · TIFF"}</span>
            </label>

            <div className="hint">
              <span>ⓘ</span>
              <p>
                Upload the original 11-band, 256 × 256 GeoTIFF.
                Do not upload label masks ending in <code>_cl.tif</code> or
                confidence masks ending in <code>_conf.tif</code>.
              </p>
            </div>

            {error && <div className="error" role="alert"><X size={16} />{error}</div>}

            <button className="btn primary full" onClick={analyze} disabled={!file || busy}>
              {busy
                ? <><LoaderCircle className="spin" size={17} /> Analyzing image...</>
                : <>Run pollution analysis <ArrowRight size={17} /></>}
            </button>

            <div className="privacy">
              <ShieldCheck size={14} /> Uploads are sent to the configured API for processing.
            </div>
          </div>

          <div className="panel results">
            <div className="panelhead">
              <b className="step">02</b>
              <div><h3>Analysis results</h3><p>Predictions returned by the U-Net model.</p></div>
              <span className={result ? "badge done" : "badge"}>
                {result ? "Complete" : "Awaiting input"}
              </span>
            </div>

            {!result ? (
              <div className="empty">
                <div className="emptyart">
                  <div /><div /><span><Microscope size={31} /></span>
                </div>
                <h4>Your map will appear here</h4>
                <p>
                  Upload a valid satellite GeoTIFF and run analysis to view
                  the predicted mask, overlay, and class statistics.
                </p>
                <div className="legend">
                  <span><i style={{ background: colors["Marine Debris"] }} />Marine debris</span>
                  <span><i style={{ background: colors["Dense Sargassum"] }} />Sargassum</span>
                  <span><i style={{ background: colors["Marine Water"] }} />Marine water</span>
                </div>
              </div>
            ) : (
              <div className="resultcontent">
                <div className="detected">
                  <span><Droplets size={23} /></span>
                  <div>
                    <small>MODEL PREDICTION</small>
                    <h4>{detected}</h4>
                    <p>{result.severity ? `Severity: ${result.severity}` : "Predicted class"}</p>
                  </div>
                </div>

                <div style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                  gap: 12,
                  margin: "18px 0"
                }}>
                  {imagePanels.map((item) => (
                    <div key={item.title}>
                      <h5 style={{ margin: "0 0 8px" }}>{item.title}</h5>
                      {item.url ? (
                        <button
                          type="button"
                          onClick={() => { setZoomImage({ title: item.title, url: imageUrl(item.url!) }); setZoomLevel(1); }}
                          title="Click to zoom and inspect"
                          style={{ display: "block", width: "100%", padding: 0, border: 0, background: "transparent", cursor: "zoom-in" }}
                        >
                          <img
                            src={imageUrl(item.url)}
                            alt={item.alt}
                            style={{
                              display: "block",
                              width: "100%",
                              aspectRatio: "1 / 1",
                              objectFit: "contain",
                              borderRadius: 10,
                              border: "1px solid #dce8e5",
                              background: "#f4f8f7"
                            }}
                          />
                          <small style={{ display: "block", marginTop: 5, color: "#53736c" }}>Click image to zoom</small>
                        </button>
                      ) : (
                        <div style={{
                          aspectRatio: "1 / 1",
                          display: "grid",
                          placeItems: "center",
                          padding: 12,
                          borderRadius: 10,
                          border: "1px dashed #cbd8d5",
                          color: "#667b78",
                          textAlign: "center",
                          fontSize: 13
                        }}>
                          {item.title === "Original image"
                            ? "Original preview not returned by API"
                            : "Image unavailable from API"}
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                <div className="stats">
                  <div><small>Predicted debris area</small><b>{area}</b></div>
                  <div><small>Mean max probability</small><b>{confidence}</b></div>
                  <div>
                    <small>Dice score</small>
                    <b>{typeof result.dice === "number" ? result.dice.toFixed(3) : "—"}</b>
                  </div>
                </div>

                {result.message && <p className="resultmessage">{result.message}</p>}

                {result.class_percentages && (
                  <div className="breakdown">
                    <h5>Predicted class breakdown</h5>
                    {Object.entries(result.class_percentages)
                      .sort((a, b) => b[1] - a[1])
                      .map(([name, value]) => (
                        <div className="classrow" key={name}>
                          <div>
                            <span><i style={{ background: colors[name] || "#9aa8b6" }} />{name}</span>
                            <b>{value.toFixed(2)}%</b>
                          </div>
                          <div className="track">
                            <span style={{
                              width: `${Math.max(0, Math.min(100, value))}%`,
                              background: colors[name] || "#9aa8b6"
                            }} />
                          </div>
                        </div>
                      ))}
                  </div>
                )}

                <div className="resultactions">
                  <button
                    className="btn primary"
                    disabled={!overlayUrl}
                    onClick={() => download(overlayUrl, "oceanmind-overlay.png")}
                  >
                    <ArrowDownToLine size={16} /> Download overlay
                  </button>
                  <button
                    className="btn outline"
                    disabled={!maskUrl}
                    onClick={() => download(maskUrl, "oceanmind-mask.png")}
                  >
                    <ArrowDownToLine size={16} /> Download mask
                  </button>
                  <button
                    className="btn outline"
                    onClick={downloadReport}
                  >
                    <ArrowDownToLine size={16} /> Generate & Download Report
                  </button>
                </div>

                <p className="caveat">
                  These are experimental predictions. Mean maximum probability
                  is not validated accuracy. No Dice score is available unless
                  ground-truth labels are supplied for evaluation.
                </p>
              </div>
            )}
          </div>
        </div>
      </section>

      <section id="workflow" className="section workflowsection">
        <div className="sectionhead">
          <div>
            <div className="eyebrow"><span /> FROM IMAGERY TO INSIGHT</div>
            <h2>A clear, traceable workflow</h2>
          </div>
          <p className="sectionintro">
            Understand each stage, from satellite input to predicted map.
          </p>
        </div>

        <div className="steps">
          <article>
            <small>01 / INPUT</small>
            <span><CloudUpload /></span>
            <h3>Upload imagery</h3>
            <p>Provide a supported 11-band optical satellite GeoTIFF.</p>
          </article>
          <article>
            <small>02 / MODEL</small>
            <span><Layers3 /></span>
            <h3>Segment with U-Net</h3>
            <p>The trained model predicts one of 15 MARIDA classes per pixel.</p>
          </article>
          <article>
            <small>03 / REVIEW</small>
            <span><MapPinned /></span>
            <h3>Inspect the map</h3>
            <p>Review predicted classes and the generated mask and overlay.</p>
          </article>
          <article>
            <small>04 / EXPORT</small>
            <span><ArrowDownToLine /></span>
            <h3>Export results</h3>
            <p>Download the generated mask and color-coded overlay.</p>
          </article>
        </div>
      </section>

      <section id="about" className="notice">
        <span><ShieldCheck size={20} /></span>
        <div>
          <b>Research prototype — not an operational alert system</b>
          <p>
            Clouds, waves, satellite artefacts, and other water types can be
            misclassified. Predictions require validation against labeled data
            and expert review before environmental decisions. SAR input and
            real-world area estimates are not supported by this checkpoint.
          </p>
        </div>
      </section>

      <footer>
        <a href="#" className="brand">
          <span className="brandmark"><Waves size={20} /></span>
          <span className="brandtext">oceanmind<span>ai</span><small>MARINE INTELLIGENCE</small></span>
        </a>
        <span>Built for healthier oceans · OceanMind AI 2.0</span>
        <span className="status"><i /> Frontend preview</span>
      </footer>
      {zoomImage && (
        <div
          onClick={() => setZoomImage(null)}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 9999,
            background: "rgba(5, 18, 25, 0.92)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: 20
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: "min(95vw, 1000px)",
              height: "85vh",
              display: "flex",
              flexDirection: "column",
              gap: 12,
              color: "white"
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <strong style={{ marginRight: "auto" }}>{zoomImage.title}</strong>
              <button onClick={() => setZoomLevel(v => Math.max(1, v - 0.5))}>−</button>
              <span>{Math.round(zoomLevel * 100)}%</span>
              <button onClick={() => setZoomLevel(v => Math.min(5, v + 0.5))}>+</button>
              <button onClick={() => setZoomLevel(1)}>Reset</button>
              <button onClick={() => setZoomImage(null)}>Close ✕</button>
            </div>
            <div
              style={{
                flex: 1,
                overflow: "auto",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                border: "1px solid #53656b",
                borderRadius: 10,
                background: "#101b22"
              }}
            >
              <img
                src={zoomImage.url}
                alt={zoomImage.title}
                style={{
                  width: `${zoomLevel * 100}%`,
                  maxWidth: "none",
                  maxHeight: zoomLevel === 1 ? "100%" : "none",
                  objectFit: "contain",
                  transformOrigin: "center"
                }}
              />
            </div>
            <small>Use the + and − controls to inspect predicted regions. Click outside to close.</small>
          </div>
        </div>
      )}
    </main>
  );
}






