/**
 * content.js — Content script
 * Runs automatically on every page load.
 * Captures a screenshot and sends it to the Flask backend for analysis.
 */

const BACKEND_URL = "http://localhost:5000/analyze";

// Wrap sendMessage as a promise and retry when the service worker is still waking up.
function sendMessage(msg, { retries = 3, retryDelay = 600 } = {}) {
  return new Promise((resolve, reject) => {
    const attempt = (remaining) => {
      chrome.runtime.sendMessage(msg, (response) => {
        const err = chrome.runtime.lastError;
        if (err) {
          if (remaining > 0 && err.message.includes("Receiving end does not exist")) {
            setTimeout(() => attempt(remaining - 1), retryDelay);
          } else {
            reject(new Error(err.message));
          }
        } else {
          resolve(response);
        }
      });
    };
    attempt(retries);
  });
}

(async () => {
  try {
    const response = await sendMessage({ action: "captureTab" });
    if (!response || !response.imageData) return;

    let result;
    try {
      const res = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: response.imageData,
          url: window.location.href,
        }),
      });
      result = await res.json();
    } catch (fetchErr) {
      console.warn("[ClickbaitDetector] Backend unreachable:", fetchErr.message);
      return;
    }

    handleDetectionResult(result);
  } catch (error) {
    // Silently ignore restricted pages (chrome://, about:, etc.)
    if (!error.message.includes("Cannot access") && !error.message.includes("not in effect")) {
      console.error("[ClickbaitDetector] Unexpected error:", error);
    }
  }
})();


function handleDetectionResult(data) {
  const { label, confidence, all_detections, bounding_boxes } = data;

  // Save result for popup
  chrome.storage.session.set({ lastResult: data });

  if (label === "clickbait" || label === "phishing") {
    showAlertBanner(label, confidence);
  }

  // Draw boxes for ALL detected elements (deceptive = red, neutral = blue)
  const boxes = all_detections && all_detections.length > 0
    ? all_detections
    : (bounding_boxes || []);
  if (boxes.length > 0) {
    drawBoundingBoxes(boxes);
  }
}

// Listen for draw requests from popup.js (Re-scan button)
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "drawBoxes") {
    drawBoundingBoxes(message.boxes || []);
    sendResponse({ ok: true });
  }
});


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


// Border color per detected class
const CLASS_COLORS = {
  fake_download_button: "#d32f2f",  // red   — deceptive
  ad_banner:            "#d32f2f",  // red   — deceptive
  close_button:         "#e65100",  // orange — potentially deceptive
  Buttons:              "#1565c0",  // blue  — neutral
  "Computer-vision":    "#2e7d32",  // green — neutral
};

const CLASS_LABELS = {
  fake_download_button: "⚠ Fake Download",
  ad_banner:            "⚠ Deceptive Ad",
  close_button:         "⚠ Fake Close",
  Buttons:              "Button",
  "Computer-vision":    "UI Element",
};

function clearBoundingBoxes() {
  document.querySelectorAll(".cd-detection-box").forEach(el => el.remove());
}

function drawBoundingBoxes(boxes) {
  // Remove any boxes already on screen before drawing new ones
  clearBoundingBoxes();

  if (!boxes || boxes.length === 0) return;

  // Server returns normalized [0,1] coords — scale to viewport CSS pixels
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  boxes.forEach((box) => {
    const pxX = Math.round(box.x * vw);
    const pxY = Math.round(box.y * vh);
    const pxW = Math.round(box.w * vw);
    const pxH = Math.round(box.h * vh);
    const pct   = Math.round((box.confidence || 0) * 100);
    const color = CLASS_COLORS[box.class] || "#7b1fa2";
    const displayLabel = CLASS_LABELS[box.class] || box.class;

    // Box overlay
    const overlay = document.createElement("div");
    overlay.className = "cd-detection-box";
    overlay.style.cssText = `
      position: fixed;
      left: ${pxX}px; top: ${pxY}px;
      width: ${pxW}px; height: ${pxH}px;
      border: 3px solid ${color};
      z-index: 2147483646;
      pointer-events: none;
      background: ${color}18;
      border-radius: 3px;
      box-sizing: border-box;
      opacity: 1;
      transition: opacity 0.6s ease;
    `;

    // Label chip above the box
    const chip = document.createElement("div");
    chip.style.cssText = `
      position: absolute;
      top: -26px; left: -3px;
      background: ${color};
      color: white;
      font-size: 11px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 3px 3px 0 0;
      white-space: nowrap;
      pointer-events: none;
      letter-spacing: 0.2px;
    `;
    chip.textContent = `${displayLabel} ${pct}%`;
    overlay.appendChild(chip);
    document.body.appendChild(overlay);
  });

  // Fade out at 4.5s, remove at 5s
  setTimeout(() => {
    document.querySelectorAll(".cd-detection-box").forEach(el => {
      el.style.opacity = "0";
    });
  }, 4500);

  setTimeout(clearBoundingBoxes, 5000);
}
