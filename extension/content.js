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
  "fake_button": "⚠️ Deceptive UI",
};

// Automatic scanning on page load is disabled due to Chrome's activeTab security policy.
// The scan is now manually triggered via the extension popup button.


function analyzeDOMContext(box) {
  // Center of the YOLO bounding box
  const cx = box.x + (box.w / 2);
  const cy = box.y + (box.h / 2);
  
  // Use elementsFromPoint to get the full stack of elements under the cursor.
  // This bypasses invisible overlays and lets us scan the parent containers!
  const elements = document.elementsFromPoint(cx, cy);
  if (!elements || elements.length === 0) return 0;

  let score = 0; 

  // Strict regex to prevent accidental matches (e.g., 'ad' matching 'download')
  const adRegex = /\b(ad|ads|advert|advertisement|sponsored|taboola|outbrain|mgid|revcontent|google_ads|adsbygoogle|doubleclick|adnxs|banner)\b/i;
  const realRegex = /\b(login|auth|submit|cart|checkout|search|nav|header|menu|footer)\b/i;
  const fakeTextRegex = /\b(download|play|start|claim|free|click here|install|watch now)\b/i;
  const realTextRegex = /\b(submit|log in|login|sign up|save|search|add to cart|subscribe|cancel)\b/i;

  // Crawl through the elements at this exact pixel coordinate
  for (let el of elements) {
    const tag = el.tagName.toLowerCase();
    
    // Stop scanning if we reach the root document
    if (tag === 'body' || tag === 'html') break;

    // 1. Tag Name Heuristics (Ad Networks)
    if (tag === 'ins' || tag === 'iframe') {
      const src = (el.src || "").toLowerCase();
      const cls = (el.className || "").toString().toLowerCase();
      if (src.includes('google') || src.includes('doubleclick') || adRegex.test(cls)) {
        return 10.0; // DEFINITELY A THIRD-PARTY AD
      }
      score += 2.0; // Highly suspicious
    }

    if (tag === 'button' || tag === 'form' || tag === 'input') {
      score -= 1.0; // Native UI forms
    }

    // 2. Class and ID Heuristics
    const id = (el.id || "").toLowerCase();
    const className = (typeof el.className === 'string' ? el.className : "").toLowerCase();
    const allAttrText = id + " " + className;

    if (adRegex.test(allAttrText)) {
      return 10.0; // DEFINITELY AN AD WRAPPER
    }
    if (realRegex.test(allAttrText)) {
      score -= 2.0; // Looks like a native navigation or structural element
    }
  }

  // 3. Text and Link Heuristics (using the topmost visible element)
  const primaryEl = elements[0];
  const text = (primaryEl.innerText || primaryEl.value || "").toLowerCase();

  if (fakeTextRegex.test(text)) score += 1.5;
  if (realTextRegex.test(text)) score -= 1.5;

  const link = primaryEl.closest('a');
  if (link && link.href) {
    try {
      const url = new URL(link.href);
      if (url.hostname !== window.location.hostname && !url.href.startsWith('javascript:')) {
        score += 1.5; // External link (Suspicious for a button)
      } else {
        score -= 1.5; // Internal navigation (Legitimate)
      }
    } catch(e) {}
  }

  return score;
}

function handleDetectionResult(data) {
  let { label, confidence, bounding_boxes, detection_count } = data;

  if (bounding_boxes && bounding_boxes.length > 0) {
    // ── Secret DOM Fusion Layer ───────────────────────────────────────
    // Fuse the Backend's pure CV score with the Frontend's DOM analysis
    bounding_boxes = bounding_boxes.map(box => {
      const domScore = analyzeDOMContext(box);
      // Combine them: cv_score is usually 0.0 to 1.0. We weight the DOM heavily.
      const combinedScore = box.cv_score + (domScore * 0.5);
      
      // Override the element type based on the final fused mathematical score
      box.type = combinedScore > 0.40 ? "fake" : "real";
      return box;
    });

    // Update the data payload so the popup gets the corrected metrics
    data.bounding_boxes = bounding_boxes;
    const fakeBoxes = bounding_boxes.filter(b => b.type === "fake");
    const fakeCount = fakeBoxes.length;
    data.detection_count = fakeCount;
    data.label = fakeCount > 0 ? "clickbait" : "clean";

    // Save result for popup
    chrome.storage.session.set({ lastResult: data });

    if (fakeCount > 0) {
      showAlertBanner(confidence, fakeCount);
    }
    // ONLY draw boxes for fake/clickbait elements
    drawClickbaitBoxes(fakeBoxes);
  } else {
    // Save empty result for popup
    chrome.storage.session.set({ lastResult: data });
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
    const displayName  = CLASS_DISPLAY_NAMES[className] || `⚠️ ${className}`;
    const confPct      = ((box.confidence || 0) * 100).toFixed(0);

    // Safeguard: make sure the box is at least 15px by 15px so it's always visible
    const w = Math.max(box.w, 15);
    const h = Math.max(box.h, 15);

    // Outer wrapper
    const wrapper = document.createElement("div");
    wrapper.id = `clickbait-box-${box.x}-${box.y}`; // Use coordinates for ID to find it later
    wrapper.style.cssText = `
      position: absolute;
      left: ${box.x + window.scrollX}px;
      top: ${box.y + window.scrollY}px;
      width: ${w}px;
      height: ${h}px;
      z-index: 2147483640;
      pointer-events: none;
      transition: transform 0.3s ease;
    `;

    // The border box (Red only)
    const borderBox = document.createElement("div");
    borderBox.style.cssText = `
      position: absolute;
      inset: 0;
      border: 3px solid #e53935;
      border-radius: 8px;
      background: rgba(229, 57, 53, 0.06);
      box-shadow: 0 0 0 1px rgba(229, 57, 53, 0.3),
                  inset 0 0 0 1px rgba(229, 57, 53, 0.1);
      animation: clickbait-pulse 2s ease-in-out infinite;
    `;

    // Label badge at the top of the box
    const label = document.createElement("div");
    label.style.cssText = `
      position: absolute;
      top: -26px;
      left: 0px;
      background: #e53935;
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

// Listen for manual re-scans triggered from the popup or scroll actions
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "drawBoxes" && message.data) {
    handleDetectionResult(message.data);
  } else if (message.action === "scrollToBox" && message.box) {
    const box = message.box;
    // Scroll the window so the box is centered
    window.scrollTo({
      left: box.x + window.scrollX - 50,
      top: box.y + window.scrollY - window.innerHeight / 2,
      behavior: 'smooth'
    });

    // Find the box we drew and flash it
    const boxId = `clickbait-box-${box.x}-${box.y}`;
    const boxEl = document.getElementById(boxId);
    
    if (boxEl) {
      boxEl.style.zIndex = "2147483647"; // Bring to very front
      boxEl.style.transform = "scale(1.2)"; // Pulse big
      
      // Flash the border white
      const borderDiv = boxEl.children[0];
      if (borderDiv) {
        borderDiv.style.borderColor = "white";
        borderDiv.style.background = "rgba(255,255,255,0.3)";
      }
      
      setTimeout(() => {
        boxEl.style.transform = "scale(1.0)";
        if (borderDiv) {
          borderDiv.style.borderColor = "#e53935"; // Back to red
          borderDiv.style.background = "rgba(229, 57, 53, 0.06)";
        }
      }, 500);
    }
  }
});
