/**
 * background.js — Background service worker
 * Listens for screenshot capture requests from content.js
 */

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "captureTab") {
    chrome.tabs.captureVisibleTab(null, { format: "png" }, (imageData) => {
      if (chrome.runtime.lastError) {
        console.error("[ClickbaitDetector] Capture error:", chrome.runtime.lastError.message);
        sendResponse({ imageData: null, error: chrome.runtime.lastError.message });
      } else {
        sendResponse({ imageData });
      }
    });
    return true; // Required to keep the message channel open for async response
  }
});
