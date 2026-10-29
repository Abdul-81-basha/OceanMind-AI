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
            <small>01 / INPUT</small><span><CloudUpload /></span>
            <h3>Upload imagery</h3>
            <p>Provide a supported 11-band optical satellite GeoTIFF.</p>
          </article>
          <article>
            <small>02 / MODEL</small><span><Layers3 /></span>
            <h3>Segment with U-Net</h3>
            <p>The trained model predicts one of 15 MARIDA classes per pixel.</p>
          </article>
          <article>
            <small>03 / REVIEW</small><span><MapPinned /></span>
            <h3>Inspect the map</h3>
            <p>Review predicted classes and the generated mask and overlay.</p>
          </article>
          <article>
            <small>04 / EXPORT</small><span><ArrowDownToLine /></span>
            <h3>Export results</h3>
            <p>Download the generated mask and color-coded overlay.</p>
          </article>
        </div>
      </section>

      <section id="about" className="notice">
        <span><ShieldCheck size={20} /></span>
        <div>
          <b>Research prototype â€” not an operational alert system</b>
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
        <span>Built for healthier oceans Â· OceanMind AI 2.0</span>
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






