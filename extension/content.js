/**
 * content.js — Content script (Rebuilt for Element-Level Detection)
 * 
 * On page load:
 *   1. Captures a screenshot of the current tab
 *   2. Sends it to the Flask backend (YOLOv8 + CV detector)
 *   3. Draws red rounded boxes around detected clickbait elements
 *   4. Shows a class label badge and confidence on each box
 */

const BACKEND_URL = "http://localhost:5000/analyze";

// Class label → display name mapping
const CLASS_DISPLAY_NAMES = {
  "Buttons":              "⚠️ Deceptive Button",
  "Computer-vision":      "🎯 Deceptive Element",
  "ad_banner":            "📢 Ad Banner",
  "close_button":         "❌ Fake Close Button",
  "fake_download_button": "⬇️ Fake Download",
};

// Automatic scanning on page load is disabled due to Chrome's activeTab security policy.
// The scan is now manually triggered via the extension popup button.


function handleDetectionResult(data) {
  const { label, confidence, bounding_boxes, detection_count } = data;

  // Save result for popup
  chrome.storage.session.set({ lastResult: data });

  if (bounding_boxes && bounding_boxes.length > 0) {
    const fakeCount = bounding_boxes.filter(b => b.type === "fake").length;
    if (fakeCount > 0) {
      showAlertBanner(confidence, fakeCount);
    }
    drawClickbaitBoxes(bounding_boxes);
  }
}


function showAlertBanner(confidence, count) {
  if (document.getElementById("clickbait-alert-banner")) return;

  const banner = document.createElement("div");
  banner.id = "clickbait-alert-banner";
  banner.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; z-index: 2147483647;
    background: linear-gradient(135deg, #b71c1c, #e53935);
    color: white; padding: 10px 20px; font-size: 14px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    text-align: center; box-shadow: 0 3px 12px rgba(0,0,0,0.4);
    display: flex; align-items: center; justify-content: center; gap: 12px;
    border-bottom: 2px solid rgba(255,255,255,0.3);
  `;

  const pct = (confidence * 100).toFixed(1);
  banner.innerHTML = `
    <span style="font-size:18px">🎯</span>
    <span>
      <strong>CLICKBAIT DETECTED</strong> — 
      ${count} deceptive element${count !== 1 ? "s" : ""} found 
      <span style="background:rgba(255,255,255,0.25);padding:2px 8px;border-radius:10px;margin-left:6px;font-size:12px">
        ${pct}% confidence
      </span>
    </span>
    <button onclick="this.parentElement.remove()" style="
      margin-left: auto; background: rgba(255,255,255,0.2);
      border: 1px solid rgba(255,255,255,0.4); color: white; 
      padding: 4px 12px; border-radius: 20px; cursor: pointer; 
      font-size: 12px; transition: background 0.2s;"
      onmouseover="this.style.background='rgba(255,255,255,0.35)'"
      onmouseout="this.style.background='rgba(255,255,255,0.2)'">
      Dismiss ✕
    </button>
  `;
  document.body.prepend(banner);
}


function drawClickbaitBoxes(boxes) {
  // Remove any existing boxes first
  document.querySelectorAll("[id^='clickbait-box-']").forEach(el => el.remove());

  boxes.forEach((box, idx) => {
    const className    = box.class_name || "Unknown";
    const isFake       = box.type === "fake";
    const displayName  = isFake ? (CLASS_DISPLAY_NAMES[className] || `⚠️ ${className}`) : "✅ Real Element";
    const confPct      = ((box.confidence || 0) * 100).toFixed(0);
    
    // Theme colors based on type
    const colorHex  = isFake ? "#e53935" : "#4caf50";
    const colorRgba = isFake ? "rgba(229, 57, 53," : "rgba(76, 175, 80,";
    const anim      = isFake ? "clickbait-pulse 2s ease-in-out infinite" : "none";

    // Outer wrapper
    const wrapper = document.createElement("div");
    wrapper.id = `clickbait-box-${idx}`;
    wrapper.style.cssText = `
      position: absolute;
      left: ${box.x + window.scrollX}px;
      top: ${box.y + window.scrollY}px;
      width: ${box.w}px;
      height: ${box.h}px;
      z-index: 2147483640;
      pointer-events: none;
    `;

    // The border box
    const borderBox = document.createElement("div");
    borderBox.style.cssText = `
      position: absolute;
      inset: 0;
      border: 3px solid ${colorHex};
      border-radius: 8px;
      background: ${colorRgba} 0.06);
      box-shadow: 0 0 0 1px ${colorRgba} 0.3),
                  inset 0 0 0 1px ${colorRgba} 0.1);
      animation: ${anim};
    `;

    // Label badge at the top of the box
    const label = document.createElement("div");
    label.style.cssText = `
      position: absolute;
      top: -26px;
      left: 0px;
      background: ${colorHex};
      color: white;
      padding: 3px 8px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 11px;
      font-weight: 600;
      border-radius: 4px 4px 0 0;
      white-space: nowrap;
      pointer-events: none;
      box-shadow: 0 -2px 6px rgba(0,0,0,0.2);
      letter-spacing: 0.3px;
    `;
    label.textContent = `${displayName}  ${confPct}%`;

    // Corner accent markers
    const corners = [
      { top: "-3px", left: "-3px", borderStyle: "solid transparent transparent solid" },
      { top: "-3px", right: "-3px", borderStyle: "solid solid transparent transparent" },
      { bottom: "-3px", left: "-3px", borderStyle: "transparent transparent solid solid" },
      { bottom: "-3px", right: "-3px", borderStyle: "transparent solid solid transparent" },
    ];
    corners.forEach(pos => {
      const corner = document.createElement("div");
      const posStr = Object.entries(pos)
        .filter(([k]) => !k.startsWith("border"))
        .map(([k, v]) => `${k}:${v}`)
        .join(";");
      corner.style.cssText = `
        position: absolute;
        ${posStr};
        width: 10px; height: 10px;
        border: 3px ${pos.borderStyle.includes("transparent transparent transparent solid") 
          ? "none" : "solid"};
        border-color: #ff1744 transparent transparent transparent;
        pointer-events: none;
      `;
    });

    wrapper.appendChild(borderBox);
    wrapper.appendChild(label);
    document.body.appendChild(wrapper);
  });

  // Inject pulse animation once
  if (!document.getElementById("clickbait-keyframes")) {
    const style = document.createElement("style");
    style.id = "clickbait-keyframes";
    style.textContent = `
      @keyframes clickbait-pulse {
        0%, 100% { box-shadow: 0 0 0 1px rgba(229,57,53,0.3), inset 0 0 0 1px rgba(229,57,53,0.1); }
        50%       { box-shadow: 0 0 0 3px rgba(229,57,53,0.5), inset 0 0 0 1px rgba(229,57,53,0.2); }
      }
    `;
    document.head.appendChild(style);
  }
}

// Listen for manual re-scans triggered from the popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "drawBoxes" && message.data) {
    handleDetectionResult(message.data);
  }
});
