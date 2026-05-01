/**
 * content.js — Content script
 * Runs automatically on every page load.
 * Captures a screenshot and sends it to the Flask backend for analysis.
 */

const BACKEND_URL = "http://localhost:5000/analyze";

(async () => {
  try {
    // Ask background service worker to capture the current tab
    chrome.runtime.sendMessage({ action: "captureTab" }, async (response) => {
      if (chrome.runtime.lastError) {
        console.error("[ClickbaitDetector]", chrome.runtime.lastError.message);
        return;
      }
      if (!response || !response.imageData) return;

      // Send screenshot to Flask backend
      let result;
      try {
        const res = await fetch(BACKEND_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: response.imageData,  // base64 PNG data URL
            url: window.location.href,
          }),
        });
        result = await res.json();
      } catch (fetchErr) {
        console.warn("[ClickbaitDetector] Backend unreachable:", fetchErr.message);
        return;
      }

      handleDetectionResult(result);
    });
  } catch (error) {
    console.error("[ClickbaitDetector] Unexpected error:", error);
  }
})();


function handleDetectionResult(data) {
  const { label, confidence, bounding_boxes } = data;

  // Save result for popup
  chrome.storage.session.set({ lastResult: data });

  if (label === "clickbait" || label === "phishing") {
    showAlertBanner(label, confidence);
    if (bounding_boxes && bounding_boxes.length > 0) {
      drawBoundingBoxes(bounding_boxes);
    }
  }
}


function showAlertBanner(label, confidence) {
  // Don't add duplicate banners
  if (document.getElementById("clickbait-alert-banner")) return;

  const banner = document.createElement("div");
  banner.id = "clickbait-alert-banner";
  banner.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; z-index: 2147483647;
    background: ${label === "phishing" ? "#d32f2f" : "#e65100"};
    color: white; padding: 12px 24px; font-size: 15px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    display: flex; align-items: center; justify-content: center; gap: 12px;
  `;

  const icon  = label === "phishing" ? "🚨" : "⚠️";
  const pct   = (confidence * 100).toFixed(1);
  const msg   = label === "phishing"
    ? `This page may be a PHISHING site (${pct}% confidence)`
    : `This page may contain CLICKBAIT elements (${pct}% confidence)`;

  banner.innerHTML = `
    <span style="font-size:20px">${icon}</span>
    <span><strong>WARNING:</strong> ${msg}</span>
    <button onclick="this.parentElement.remove()" style="
      margin-left: auto; background: rgba(255,255,255,0.25);
      border: none; color: white; padding: 4px 10px;
      border-radius: 4px; cursor: pointer; font-size: 13px;">
      Dismiss
    </button>
  `;
  document.body.prepend(banner);
}


function drawBoundingBoxes(boxes) {
  boxes.forEach((box, idx) => {
    const overlay = document.createElement("div");
    overlay.id = `clickbait-box-${idx}`;
    overlay.style.cssText = `
      position: fixed;
      left: ${box.x}px; top: ${box.y}px;
      width: ${box.w}px; height: ${box.h}px;
      border: 3px solid #d32f2f; z-index: 2147483646;
      pointer-events: none;
      background: rgba(211, 47, 47, 0.08);
      border-radius: 2px;
    `;
    document.body.appendChild(overlay);
  });
}
