# Tactus

FastAPI backend for an ESP32-CAM accessibility device.  
Receives images via HTTP, runs Azure AI Vision, and returns JSON descriptions or OCR text.

## Project Structure

```
project/
├── main.py           # FastAPI app and endpoints
├── azure_vision.py   # Azure AI Vision helper (describe + OCR)
├── config.py         # Environment variable loading
├── requirements.txt
├── .env.example      # Copy to .env and fill in your credentials
├── README.md
└── uploads/          # Debug copies of received images (auto-created)
```

---

## 1. Install Dependencies

```bash
cd project
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

## 2. Configure Azure Credentials

```bash
cp .env.example .env
```

Edit `.env`:

```
VISION_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com
VISION_KEY=your_32_character_key_here
```

> **No Azure account?** Leave `.env.example` as-is (or don't create `.env`).  
> The server automatically falls back to **mock output** so you can test without credentials.

---

## 3. Run the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000` to confirm it's running. You should see:

```json
{ "status": "ok", "service": "Accessibility Vision API", "mock_mode": true }
```

---

## 4. Test with curl

### Describe mode (scene description)

```bash
curl -X POST http://localhost:8000/process-image \
  -F "imageFile=@/path/to/photo.jpg" \
  -F "mode=describe"
```

**Sample response:**

```json
{
  "success": true,
  "mode": "describe",
  "filename": "20241201_143022_a1b2c3_photo.jpg",
  "caption": "a bag of potato chips on a wooden table",
  "text": "a bag of potato chips on a wooden table",
  "lines": [],
  "error": null
}
```

---

### Read mode (OCR / text extraction)

```bash
curl -X POST http://localhost:8000/process-image \
  -F "imageFile=@/path/to/menu.jpg" \
  -F "mode=read"
```

**Sample response:**

```json
{
  "success": true,
  "mode": "read",
  "filename": "20241201_143055_d4e5f6_menu.jpg",
  "caption": "",
  "text": "LAYS Classic Potato Chips\nNET WT 8 OZ (226g)\nCalories 160",
  "lines": [
    "LAYS Classic Potato Chips",
    "NET WT 8 OZ (226g)",
    "Calories 160"
  ],
  "error": null
}
```

---

## 5. Interactive API Docs

FastAPI includes built-in docs you can use in a browser:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 6. Swapping Mock Output for Real Azure Output

All CV logic lives in `azure_vision.py`.

| Function | What it calls | Mock fallback |
|---|---|---|
| `describe_image(bytes)` | Azure caption API | `_mock_describe()` |
| `read_image(bytes)` | Azure OCR/read API | `_mock_read()` |

When `VISION_ENDPOINT` and `VISION_KEY` are present in `.env`, the server automatically uses real Azure calls. If either is missing, it silently uses the mock functions at the bottom of `azure_vision.py`.

To customize mock output for demos, edit `_mock_describe()` and `_mock_read()` in `azure_vision.py`.

---

## 7. ESP32-CAM Integration

Your ESP32-CAM should POST to:

```
POST http://<your-laptop-ip>:8000/process-image
Content-Type: multipart/form-data
```

Form fields:
- `imageFile` — the JPEG image data
- `mode` — `describe` or `read`

Make sure your laptop firewall allows port 8000, and both devices are on the same network.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `mock_mode: true` but you have credentials | Check that `.env` exists (not just `.env.example`) and `VISION_ENDPOINT`/`VISION_KEY` are set |
| `504 Timeout` | Increase `REQUEST_TIMEOUT` in `.env` or check Azure resource region |
| `400 Invalid mode` | ESP32-CAM must send `mode=describe` or `mode=read` exactly |
| Port already in use | Change `--port 8000` to another port like `8001` |
