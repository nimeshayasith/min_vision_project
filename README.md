# min_vision_project

## Quick: Load the Chrome extension & run the demo

1. Install dependencies
```pip install -r requirements.txt```

2. Start the Flask backend (loads the trained model):

```bash
python -m src.server.app
```

3. In Chrome, open Extensions (chrome://extensions), enable **Developer mode**, click **Load unpacked**, and select the `extension/` folder from the project root.

4. Open any web page, click the extension icon, and use **Re-scan This Page** in the popup. The extension captures a screenshot and sends it to the backend.

5. Results are shown in the popup (label, confidence, probabilities). If the page is detected as `phishing` or `clickbait`, an in-page alert banner will appear.

Notes:
- Ensure the Flask server is running on `http://localhost:5000` (default). If you change the host/port update `extension/popup/popup.js` and `extension/content.js` `BACKEND_URL` accordingly.
- For local testing without a trained checkpoint, the server will start with an untrained model (see `models/checkpoints/`).