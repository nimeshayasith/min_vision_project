/**
 * popup.js — Popup controller
 * Reads last scan result from session storage and renders it.
 * Also supports manual re-scan via button.
 */

const BACKEND_URL = "http://localhost:5000/analyze";

const labelBadge      = document.getElementById("label-badge");
const confidenceFill  = document.getElementById("confidence-fill");
const confLabel       = document.getElementById("conf-label");
const probsContainer  = document.getElementById("probs");
const scanBtn         = document.getElementById("scan-btn");

// Colour maps for each class
const BADGE_CLASS = {
  legitimate: "badge-legitimate",
  phishing:   "badge-phishing",
  clickbait:  "badge-clickbait",
};
const FILL_CLASS = {
  legitimate: "fill-legitimate",
  phishing:   "fill-phishing",
  clickbait:  "fill-clickbait",
};

function renderResult(result) {
  if (!result) {
    labelBadge.textContent = "No result yet";
    return;
  }

  const { label, confidence, probabilities } = result;
  const pct = (confidence * 100).toFixed(1);

  // Badge
  labelBadge.className = `label-badge ${BADGE_CLASS[label] || "badge-unknown"}`;
  labelBadge.textContent = label || "unknown";

  // Confidence bar
  confidenceFill.className = `confidence-fill ${FILL_CLASS[label] || "fill-unknown"}`;
  confidenceFill.style.width = `${(confidence * 100).toFixed(0)}%`;
  confLabel.textContent = `Confidence: ${pct}%`;

  // Probability breakdown
  probsContainer.innerHTML = "";
  if (probabilities) {
    Object.entries(probabilities).forEach(([cls, prob]) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `<span>${cls}</span><span>${(prob * 100).toFixed(1)}%</span>`;
      probsContainer.appendChild(row);
    });
  }
}

// Load last result from session storage
chrome.storage.session.get(["lastResult"], ({ lastResult }) => {
  renderResult(lastResult);
});

// Re-scan button: capture current tab and send to backend
scanBtn.addEventListener("click", async () => {
  scanBtn.textContent = "⏳ Scanning...";
  scanBtn.disabled = true;

  chrome.tabs.query({ active: true, currentWindow: true }, async ([tab]) => {
    chrome.tabs.captureVisibleTab(null, { format: "png" }, async (imageData) => {
      if (!imageData) {
        scanBtn.textContent = "❌ Capture failed";
        scanBtn.disabled = false;
        return;
      }
      try {
        const res = await fetch(BACKEND_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: imageData, url: tab.url }),
        });
        const result = await res.json();
        chrome.storage.session.set({ lastResult: result });
        renderResult(result);

        // Send all detected elements to content.js to draw boxes on the page
        const boxes = result.all_detections && result.all_detections.length > 0
          ? result.all_detections
          : (result.bounding_boxes || []);

        if (boxes.length > 0) {
          chrome.tabs.sendMessage(tab.id, { action: "drawBoxes", boxes }, () => {
            void chrome.runtime.lastError; // suppress "Receiving end does not exist" on restricted pages
          });
          confLabel.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%  — highlighting ${boxes.length} element(s) for 5s`;
        }
      } catch (e) {
        labelBadge.textContent = "Server offline";
        confLabel.textContent = "Make sure Flask server is running (python -m src.server.app)";
      }
      scanBtn.textContent = "🔄 Re-scan This Page";
      scanBtn.disabled = false;
    });
  });
});
