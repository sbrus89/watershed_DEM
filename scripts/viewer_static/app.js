(function () {
  "use strict";

  const SOURCE_LABELS = {
    nodata: "no data", s1m: "USGS Seamless 1-m DEM (S1M)",
    dem1m_projects: "Project-based 1-m DEM", dem13: "USGS 1/3\" DEM",
    ehydro: "USACE eHydro (multibeam)", eaarlb: "EAARL-B lidar bathy",
    cudem19: "NOAA CUDEM 1/9\"", coned: "NJ/DE CoNED 1-m topobathy",
    filled: "gap-filled", nos_grids: "NOS hydrographic surveys",
    burn: "channel burn",
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
    const kind = `${activeTab}.png`; // hillshade | depth | burndepth
    const img = new Image();
    img.onload = () => {
      panelImage.src = img.src;
      panelImage.style.visibility = "visible";
      panelImageStatus.textContent = "";
    };
    img.onerror = () => {
      panelImageStatus.textContent = activeTab === "burndepth"
        ? "nothing burned on this tile" : "no data for this tile";
    };
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
    if (meta.burn) {
      document.getElementById("burnControls").hidden = false;
      document.getElementById("burnTab").hidden = false;
      document.getElementById("gaugeControls").hidden = false;
    }
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

  // ---- burn-depth layer (burned product only) ------------------------------
  //
  // Original (merged) minus burned elevation, from the per-tile burn-depth
  // sidecars; block-max shrunk server-side so narrow channels stay visible
  // when zoomed out (where they read shallower: the overviews are averages).

  const burnToggle = document.getElementById("burnToggle");
  const burnOpacityRow = document.getElementById("burnOpacityRow");
  const burnOpacitySlider = document.getElementById("burnOpacitySlider");
  const burnOpacityLabel = document.getElementById("burnOpacityLabel");
  const burnLegendEl = document.getElementById("burnLegend");
  let burnLayer = null;
  let burnLegendPromise = null;

  burnToggle.addEventListener("change", () => {
    burnOpacityRow.hidden = !burnToggle.checked;
    if (!burnToggle.checked) {
      if (burnLayer) map.removeLayer(burnLayer);
      burnLegendEl.hidden = true;
      return;
    }
    if (!burnLegendPromise) {
      burnLegendPromise = fetch("/api/burndepth_legend").then((r) => {
        if (!r.ok) throw new Error("burn depth not available");
        return r.json();
      }).then((info) => {
        const stops = info.stops.map(([pct, color]) => `${color} ${pct}%`).join(", ");
        document.getElementById("burnGradient").style.background =
          `linear-gradient(to right, ${stops})`;
      });
    }
    burnLegendPromise.then(() => {
      burnLegendEl.hidden = false;
      if (!burnLayer) {
        burnLayer = L.tileLayer("/api/burndepth_tile/{z}/{x}/{y}.png", {
          opacity: burnOpacitySlider.value / 100, maxZoom: 18, maxNativeZoom: 16,
        });
      }
      burnLayer.addTo(map);
    }).catch(() => {
      burnToggle.checked = false;
      burnOpacityRow.hidden = true;
      window.alert("Burn-depth layer not available -- no _burndepth.tif sidecars found.");
    });
  });

  burnOpacitySlider.addEventListener("input", () => {
    burnOpacityLabel.textContent = `${burnOpacitySlider.value}%`;
    if (burnLayer) burnLayer.setOpacity(burnOpacitySlider.value / 100);
  });

  // ---- USGS gauge comparison (burned product only) ----------------------
  //
  // Points from <root>/qa/gauge_compare.geojson (scripts/burn_gauge_compare.py):
  // burn model mean depth D at the snapped channel sample vs the gauge's
  // observed mean depth (A/W) at mean annual flow. Colour = log ratio,
  // blue = burn too shallow, red = too deep; hollow grey = not burned there.

  const gaugeToggle = document.getElementById("gaugeToggle");
  const gaugeLegendEl = document.getElementById("gaugeLegend");
  let gaugeLayer = null;
  const RAMP = ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"];

  function hexToRgb(h) { return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)); }
  function ratioColor(r) {
    // log2 ratio clipped to [-1, 1] (= 1/2x .. 2x) along the 5-stop ramp
    const t = (Math.max(-1, Math.min(1, Math.log2(r))) + 1) / 2 * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i;
    const a = hexToRgb(RAMP[i]), b = hexToRgb(RAMP[i + 1]);
    return "rgb(" + a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",") + ")";
  }
  function fmt(v, d, unit) { return v === null || v === undefined ? "&ndash;" : `${Number(v).toFixed(d)}${unit || ""}`; }

  function gaugePopup(p) {
    const status = { burned: "burned", not_lowered: `not lowered (${p.prov})`,
                     no_network: "no burn network within snap distance",
                     no_model_depth: "burn spec has no depth here (D = NaN): not burned" }[p.status] || p.status;
    const rows = [
      ["observed mean depth", fmt(p.d_obs, 2, " m")],
      ["burn model D", fmt(p.model_D, 2, " m") + (p.tier ? ` (tier ${p.tier})` : "")],
      ["model / observed", p.ratio ? fmt(p.ratio, 2, "&times;") : "&ndash;"],
      ["observed width", fmt(p.w_obs, 1, " m")],
      ["burn width (tier 2)", fmt(p.model_W, 1, " m")],
      ["deepest cut within 30 m", fmt(p.max_cut, 2, " m")],
      ["snapped", fmt(p.snap_m, 0, " m away")],
      ["drainage area", fmt(p.da_km2, 0, " km&sup2;")],
      ["measurements near mean Q", fmt(p.n_win, 0)],
      ["status", escapeHtml(status)],
    ];
    return `<b>${escapeHtml(p.name || "")}</b><br>USGS ${escapeHtml(p.site_no)} &middot; HUC ${escapeHtml(p.huc4)}` +
      `<table class="gaugePopup">${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>`;
  }

  gaugeToggle.addEventListener("change", () => {
    if (!gaugeToggle.checked) {
      if (gaugeLayer) map.removeLayer(gaugeLayer);
      gaugeLegendEl.hidden = true;
      return;
    }
    const ready = gaugeLayer ? Promise.resolve() :
      fetch("/api/gauges.geojson").then((r) => {
        if (!r.ok) throw new Error("no gauge comparison");
        return r.json();
      }).then((fc) => {
        gaugeLayer = L.geoJSON(fc, {
          pointToLayer: (f, ll) => {
            const p = f.properties, ok = p.status === "burned" && p.ratio;
            return L.circleMarker(ll, ok
              ? { radius: 6, color: "#222", weight: 1, fillColor: ratioColor(p.ratio), fillOpacity: 0.95 }
              : { radius: 5, color: "#888", weight: 2, fill: false });
          },
          onEachFeature: (f, layer) => layer.bindPopup(gaugePopup(f.properties)),
        });
      });
    ready.then(() => {
      gaugeLayer.addTo(map);
      gaugeLegendEl.hidden = false;
    }).catch(() => {
      gaugeToggle.checked = false;
      window.alert("Gauge layer not available -- run scripts/burn_gauge_compare.py first.");
    });
  });

  // ---- basin boundary -----------------------------------------------------
  //
  // The HUC-4 0204 watershed polygon (stage 01's dissolved WBD boundary,
  // config/tile discovery's own basin clip) -- a plain reference outline,
  // not clickable and carrying no per-feature panel like the tile lattice.

  fetch("/api/basin_boundary.geojson").then((r) => {
    if (!r.ok) throw new Error("no basin boundary");
    return r.json();
  }).then((fc) => {
    L.geoJSON(fc, {
      interactive: false,
      style: { color: "#ffe066", weight: 2.5, fill: false },
    }).addTo(map);
  }).catch(() => {});

  // ---- load tiles and fit map --------------------------------------------

  fetch("/api/tiles.geojson").then((r) => r.json()).then((fc) => {
    tileLayerGroup.addData(fc);
    const b = tileLayerGroup.getBounds();
    if (b.isValid()) map.fitBounds(b.pad(0.05));
  });
})();
