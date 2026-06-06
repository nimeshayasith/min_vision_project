/**
 * popup.js — Popup controller (Updated for YOLOv8 element-level detection)
 * Reads last scan result from session storage and renders it.
 * Also supports manual re-scan via button.
 */

const BACKEND_URL = "http://localhost:5000/analyze";

const labelBadge     = document.getElementById("label-badge");
const confidenceFill = document.getElementById("confidence-fill");
const confLabel      = document.getElementById("conf-label");
const probsContainer = document.getElementById("probs");
const scanBtn        = document.getElementById("scan-btn");

// Class display names
const CLASS_ICONS = {
  "Buttons":              "⚠️",
  "Computer-vision":      "🎯",
  "ad_banner":            "📢",
  "close_button":         "❌",
  "fake_download_button": "⬇️",
};

const BADGE_CLASS = {
  clickbait: "badge-clickbait",
  clean:     "badge-legitimate",
};
const FILL_CLASS = {
  clickbait: "fill-clickbait",
  clean:     "fill-legitimate",
};

function renderResult(result) {
  if (!result) {
    labelBadge.textContent = "No scan yet";
    confLabel.textContent  = "Open a webpage to trigger automatic scan";
    return;
  }

  const { label, confidence, bounding_boxes, detection_count } = result;
  const pct = (confidence * 100).toFixed(1);

  // Badge
  labelBadge.className   = `label-badge ${BADGE_CLASS[label] || "badge-unknown"}`;
  labelBadge.textContent = label === "clickbait" ? "⚠️ CLICKBAIT" : "✅ CLEAN";

  // Confidence bar
  confidenceFill.className  = `confidence-fill ${FILL_CLASS[label] || "fill-unknown"}`;
  confidenceFill.style.width = `${(confidence * 100).toFixed(0)}%`;
  confLabel.textContent     = `Confidence: ${pct}%`;

  // Probability breakdown / detection list
  probsContainer.innerHTML = "";

  if (bounding_boxes && bounding_boxes.length > 0) {
    probsContainer.style.display = "block";

    const header = document.createElement("div");
    header.className   = "prob-row";
    header.style.cssText = "font-weight:600;color:#e2e8f0;margin-bottom:4px;";
    header.innerHTML = `<span>🎯 Detected Elements</span><span style="background:rgba(239,68,68,0.2);color:#fca5a5">${detection_count || bounding_boxes.length} found</span>`;
    probsContainer.appendChild(header);

    bounding_boxes.forEach((box) => {
      const icon = CLASS_ICONS[box.class_name] || "⚠️";
      const row  = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span>${icon} ${box.class_name || "Unknown"}</span>
        <span>${((box.confidence || 0) * 100).toFixed(1)}%</span>
      `;
      probsContainer.appendChild(row);
    });
  } else if (label === "clean") {
    probsContainer.style.display = "block";
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `<span>✅ No clickbait elements detected on this page</span>`;
    probsContainer.appendChild(row);
  } else {
    probsContainer.style.display = "none";
  }
}

// Load last result from session storage
chrome.storage.session.get(["lastResult"], ({ lastResult }) => {
  renderResult(lastResult);
});

// Re-scan button: capture current tab and send to backend
scanBtn.addEventListener("click", async () => {
  scanBtn.textContent = "⏳ Scanning...";
  scanBtn.disabled    = true;

  chrome.tabs.query({ active: true, currentWindow: true }, async ([tab]) => {
    chrome.tabs.captureVisibleTab(null, { format: "png" }, async (imageData) => {
      if (!imageData) {
        scanBtn.textContent = "❌ Capture failed";
        scanBtn.disabled    = false;
        return;
      }
      try {
        const res = await fetch(BACKEND_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: imageData,
            url:   tab.url,
            dpr:   window.devicePixelRatio || 1,
          }),
        });
        const result = await res.json();
        chrome.storage.session.set({ lastResult: result });
        renderResult(result);
        
        // Tell the content script on the page to draw the red boxes
        chrome.tabs.sendMessage(tab.id, { action: "drawBoxes", data: result }).catch(() => {});
      } catch (e) {
        labelBadge.textContent = "Server offline";
        confLabel.textContent  = "Run: python -m src.server.app";
      }
      scanBtn.textContent = "🔄 Re-scan This Page";
      scanBtn.disabled    = false;
    });
  });
});
