(function () {
  "use strict";

  const SOURCE_LABELS = {
    nodata: "no data", s1m: "USGS Seamless 1-m DEM (S1M)",
    dem1m_projects: "Project-based 1-m DEM", dem13: "USGS 1/3\" DEM",
    ehydro: "USACE eHydro (multibeam)", eaarlb: "EAARL-B lidar bathy",
    cudem19: "NOAA CUDEM 1/9\"", coned: "NJ/DE CoNED 1-m topobathy",
    filled: "gap-filled",
  };

  const map = L.map("map", { zoomControl: true });

  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 18, attribution: "Esri World Imagery" }
  ).addTo(map);

  const tileLayerGroup = L.geoJSON(null, { style: styleForFeature, onEachFeature }).addTo(map);
  let selectedLayer = null;
  let depthLayer = null;
  let depthLegendPromise = null;

  function styleForFeature(feature) {
    const p = feature.properties;
    if (!p.has_final_tile) {
      return { color: "#999", weight: 1, dashArray: "4,3", fillOpacity: 0.05 };
    }
    const flagged = (p.seam_flags && p.seam_flags.length) ||
                    (p.size_mismatches && p.size_mismatches.length);
    const badOverlap = p.land_overlap && Math.abs(p.land_overlap.median) > 0.10;
    if (flagged || badOverlap) {
      return { color: "#ff6b3b", weight: 1.6, fillOpacity: 0.06 };
    }
    return { color: "#ff3b30", weight: 1, fillOpacity: 0.02 };
  }

  function onEachFeature(feature, layer) {
    layer.on("click", () => selectTile(layer, feature.properties));
  }

  function selectTile(layer, props) {
    if (selectedLayer) tileLayerGroup.resetStyle(selectedLayer);
    selectedLayer = layer;
    layer.setStyle({ color: "#2d6cdf", weight: 3, fillOpacity: 0.08 });
    layer.bringToFront();
    openPanel(props);
  }

  // ---- side panel -------------------------------------------------------

  const panel = document.getElementById("panel");
  const panelTitle = document.getElementById("panelTitle");
  const panelFlags = document.getElementById("panelFlags");
  const panelCoverage = document.getElementById("panelCoverage");
  const panelImage = document.getElementById("panelImage");
  const panelSourceOverlay = document.getElementById("panelSourceOverlay");
  const panelImageStatus = document.getElementById("panelImageStatus");
  const sourceOverlayToggle = document.getElementById("sourceOverlayToggle");
  const sourceOpacityRow = document.getElementById("sourceOpacityRow");
  const sourceOpacitySlider = document.getElementById("sourceOpacitySlider");
  const sourceOpacityLabel = document.getElementById("sourceOpacityLabel");
  let activeTab = "hillshade";
  let activeTileName = null;

  document.getElementById("panelClose").addEventListener("click", () => {
    panel.hidden = true;
    if (selectedLayer) { tileLayerGroup.resetStyle(selectedLayer); selectedLayer = null; }
  });

  document.querySelectorAll(".tabBtn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tabBtn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeTab = btn.dataset.tab;
      loadTileImage();
    });
  });

  function openPanel(props) {
    activeTileName = props.tile;
    panel.hidden = false;
    panelTitle.textContent = props.tile;

    const flags = [];
    if (!props.has_final_tile) {
      flags.push(["err", "no final tile on disk yet"]);
    }
    if (props.land_overlap) {
      const m = props.land_overlap.median;
      const cls = Math.abs(m) > 0.10 ? "warn" : null;
      const text = `land vs. CUDEM median offset: ${m.toFixed(3)} m (n=${props.land_overlap.n.toLocaleString()})`;
      if (cls) flags.push([cls, text]);
    }
    (props.seam_flags || []).forEach((f) =>
      flags.push(["warn", `seam flagged on ${f.edge} (${f.value.toFixed(2)} m step)`]));
    (props.size_mismatches || []).forEach((f) =>
      flags.push(["warn", `size mismatch on ${f.edge} (${f.value[0]} vs ${f.value[1]} px)`]));
    panelFlags.innerHTML = flags
      .map(([cls, text]) => `<div class="flag ${cls}">${escapeHtml(text)}</div>`)
      .join("");

    renderCoverage(props.coverage || {});
    loadTileImage();
    loadSourceOverlay();
  }

  function renderCoverage(cov) {
    const entries = Object.entries(cov).filter(([, pct]) => pct > 0)
      .sort((a, b) => b[1] - a[1]);
    if (!entries.length) {
      panelCoverage.innerHTML = "<em>no coverage data</em>";
      return;
    }
    const bar = entries.map(([name, pct]) =>
      `<span style="width:${pct}%;background:${colorFor(name)}" title="${name} ${pct}%"></span>`
    ).join("");
    const rows = entries.map(([name, pct]) => `
      <div class="covRow">
        <span class="name"><span class="swatch" style="background:${colorFor(name)}"></span>${SOURCE_LABELS[name] || name}</span>
        <span>${pct.toFixed(1)}%</span>
      </div>`).join("");
    panelCoverage.innerHTML = `<div class="covBar">${bar}</div>${rows}`;
  }

  function loadTileImage() {
    if (!activeTileName) return;
    panelImageStatus.textContent = "loading…";
    panelImage.style.visibility = "hidden";
    const kind = `${activeTab}.png`; // "hillshade.png" | "depth.png"
    const img = new Image();
    img.onload = () => {
      panelImage.src = img.src;
      panelImage.style.visibility = "visible";
      panelImageStatus.textContent = "";
    };
    img.onerror = () => { panelImageStatus.textContent = "no data for this tile"; };
    img.src = `/api/tile/${encodeURIComponent(activeTileName)}/${kind}?max=1400`;
  }

  // Source provenance as a translucent layer on top of whichever base image
  // (hillshade or depth) is showing, rather than its own mutually-exclusive
  // tab -- loaded independently of activeTab so flipping base tabs doesn't
  // need a refetch, and cheap regardless since the server caches it per tile.
  function loadSourceOverlay() {
    if (!activeTileName) return;
    panelSourceOverlay.style.opacity = sourceOpacitySlider.value / 100;
    panelSourceOverlay.src = `/api/tile/${encodeURIComponent(activeTileName)}/source.png?max=1400`;
  }

  // The "error" listener can fire after the toggle is already on (a newly
  // selected tile with no provenance raster) -- it wins over the checkbox
  // either way, since there's nothing to show.
  panelSourceOverlay.addEventListener("error", () => { panelSourceOverlay.hidden = true; });

  sourceOverlayToggle.addEventListener("change", () => {
    sourceOpacityRow.hidden = !sourceOverlayToggle.checked;
    panelSourceOverlay.hidden = !sourceOverlayToggle.checked;
  });

  sourceOpacitySlider.addEventListener("input", () => {
    sourceOpacityLabel.textContent = `${sourceOpacitySlider.value}%`;
    panelSourceOverlay.style.opacity = sourceOpacitySlider.value / 100;
  });

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---- source legend ------------------------------------------------------

  let legendColors = {};
  function colorFor(name) { return legendColors[name] ? legendColors[name].color : "#888"; }

  fetch("/api/meta").then((r) => r.json()).then((meta) => {
    document.getElementById("title").textContent = `${meta.name} ${meta.version}`;
    legendColors = meta.legend;
    const legendEl = document.getElementById("legend");
    legendEl.innerHTML = Object.entries(meta.legend).map(([name, info]) => `
      <div class="swatchRow">
        <span class="swatch" style="background:${info.color}"></span>
        <span>${SOURCE_LABELS[name] || name}</span>
      </div>`).join("");
  });

  // ---- depth (elevation) layer toggle --------------------------------------
  //
  // A real XYZ tile layer read window-by-window off the basin VRT (see
  // /api/depth_tile/{z}/{x}/{y}.png), not an L.imageOverlay of the old
  // matplotlib mosaic_topo.png: that PNG's baked-in title/colorbar/axis
  // padding meant its pixel frame didn't match its geographic bounds, so it
  // never registered cleanly over the imagery. A tile layer is always
  // correctly georeferenced because each tile is windowed to its own
  // lon/lat bounds server-side.

  const depthToggle = document.getElementById("depthToggle");
  const opacityRow = document.getElementById("opacityRow");
  const opacitySlider = document.getElementById("opacitySlider");
  const opacityLabel = document.getElementById("opacityLabel");
  const depthLegendEl = document.getElementById("depthLegend");

  function ensureDepthLegend() {
    if (!depthLegendPromise) {
      depthLegendPromise = fetch("/api/depth_legend").then((r) => {
        if (!r.ok) throw new Error("depth legend not available");
        return r.json();
      }).then((info) => {
        const stops = info.stops.map(([pct, color]) => `${color} ${pct}%`).join(", ");
        document.getElementById("depthGradient").style.background =
          `linear-gradient(to right, ${stops})`;
        document.getElementById("depthLo").textContent = info.lo.toFixed(1);
        document.getElementById("depthHi").textContent = `+${info.hi.toFixed(1)}`;
        depthLegendEl.hidden = false;
        return info;
      });
    }
    return depthLegendPromise;
  }

  depthToggle.addEventListener("change", () => {
    opacityRow.hidden = !depthToggle.checked;
    if (!depthToggle.checked) {
      if (depthLayer) map.removeLayer(depthLayer);
      depthLegendEl.hidden = true;
      return;
    }
    ensureDepthLegend().then(() => {
      if (!depthLayer) {
        depthLayer = L.tileLayer("/api/depth_tile/{z}/{x}/{y}.png", {
          opacity: opacitySlider.value / 100, maxZoom: 18, maxNativeZoom: 15,
        });
      }
      depthLayer.addTo(map);
    }).catch(() => {
      depthToggle.checked = false;
      opacityRow.hidden = true;
      window.alert("Depth layer not available -- run stage 09 to build the basin VRT first.");
    });
  });

  opacitySlider.addEventListener("input", () => {
    opacityLabel.textContent = `${opacitySlider.value}%`;
    if (depthLayer) depthLayer.setOpacity(opacitySlider.value / 100);
  });

  // ---- sources (provenance) layer toggle -----------------------------------
  //
  // Same approach as the depth layer: a real XYZ tile layer read window-by-
  // window off a mosaic (see /api/source_tile/{z}/{x}/{y}.png), colorized
  // with the same palette the per-tile provenance panel already uses, so the
  // basin overlay and a selected tile's "overlay source provenance" reading
  // agree with each other.

  const sourcesToggle = document.getElementById("sourcesToggle");
  const sourcesOpacityRow = document.getElementById("sourcesOpacityRow");
  const sourcesOpacitySlider = document.getElementById("sourcesOpacitySlider");
  const sourcesOpacityLabel = document.getElementById("sourcesOpacityLabel");
  let sourcesLayer = null;
  let sourcesAvailablePromise = null;

  function ensureSourcesAvailable() {
    if (!sourcesAvailablePromise) {
      sourcesAvailablePromise = fetch("/api/source_legend").then((r) => {
        if (!r.ok) throw new Error("sources layer not available");
      });
    }
    return sourcesAvailablePromise;
  }

  sourcesToggle.addEventListener("change", () => {
    sourcesOpacityRow.hidden = !sourcesToggle.checked;
    if (!sourcesToggle.checked) {
      if (sourcesLayer) map.removeLayer(sourcesLayer);
      return;
    }
    ensureSourcesAvailable().then(() => {
      if (!sourcesLayer) {
        sourcesLayer = L.tileLayer("/api/source_tile/{z}/{x}/{y}.png", {
          opacity: sourcesOpacitySlider.value / 100, maxZoom: 18, maxNativeZoom: 15,
        });
      }
      sourcesLayer.addTo(map);
    }).catch(() => {
      sourcesToggle.checked = false;
      sourcesOpacityRow.hidden = true;
      window.alert("Sources layer not available -- no provenance rasters found yet.");
    });
  });

  sourcesOpacitySlider.addEventListener("input", () => {
    sourcesOpacityLabel.textContent = `${sourcesOpacitySlider.value}%`;
    if (sourcesLayer) sourcesLayer.setOpacity(sourcesOpacitySlider.value / 100);
  });

  // ---- load tiles and fit map --------------------------------------------

  fetch("/api/tiles.geojson").then((r) => r.json()).then((fc) => {
    tileLayerGroup.addData(fc);
    const b = tileLayerGroup.getBounds();
    if (b.isValid()) map.fitBounds(b.pad(0.05));
  });
})();
