# Tactus

FastAPI backend for an ESP32-CAM accessibility device.
Captures images via HTTP, runs computer vision, translates output to UEB Grade 2 Braille, and sends the Braille payload to physical motors on the ESP32.

## CV Backends

Set `VISION_BACKEND` in `.env` to switch between:

| Backend | describe | read | Internet required |
|---|---|---|---|
| `local` (default) | Ollama (llava-phi3) | Tesseract OCR | No |
| `azure` | Azure AI Vision (tags + objects) | Azure OCR | Yes |
| `gemini` | Google Gemini Vision | Google Gemini Vision | Yes |

---

## Project Structure

```
tactus/
├── main.py               # FastAPI app and endpoints
├── azure_vision.py       # Azure AI Vision backend
├── local_vision.py       # Local backend (Ollama + Tesseract)
├── gemini_vision.py      # Gemini API backend (requires API key)
├── braille_translator.py # UEB Grade 2 Braille translator
├── camera_bridge.py      # Polls ESP32 button, runs pipeline, sends Braille
├── config.py             # Environment variable loading
├── .env.example          # Copy to .env and fill in credentials
└── uploads/              # Debug copies of received images (auto-created)
```

---

## 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
brew install tesseract    # macOS — required for local OCR
```

---

## 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```
VISION_BACKEND=local        # or azure or gemini

# Azure (only needed if VISION_BACKEND=azure)
VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com
VISION_KEY=your_key_here

# Gemini (only needed if VISION_BACKEND=gemini)
GEMINI_API_KEY=your_gemini_key_here

# Local Ollama (only needed if VISION_BACKEND=local)
OLLAMA_MODEL=llava-phi3
```

---

## 3. Run

**Terminal 1 — Ollama (local backend only):**
```bash
ollama serve
```

**Terminal 2 — FastAPI server:**
```bash
uvicorn main:app --reload
```

**Terminal 3 — Camera bridge (connects to ESP32):**
```bash
python camera_bridge.py
```

The camera bridge polls the ESP32's `/reading` endpoint every second. When the button is pressed, it captures a frame, processes it, translates to Braille, and POSTs the payload to the ESP32's `/braille` endpoint.

---

## 4. Test with curl

### Describe mode
```bash
curl -s -X POST http://localhost:8000/process-image \
  -F "imageFile=@test_images/dog_test.jpg" \
  -F "mode=describe" | python3 -m json.tool
```

### Read mode (OCR)
```bash
curl -s -X POST http://localhost:8000/process-image \
  -F "imageFile=@test_images/test_text_image.jpg" \
  -F "mode=read" | python3 -m json.tool
```

---

## 5. Braille Translation

All text output is translated to UEB Grade 2 Braille. The response includes:

- `braille_payload` — comma-separated 6-bit cells sent to the ESP32 motors
- `braille_debug` — human-readable mapping of each character to its cell

Bit mapping per cell:
```
Index 0 = Dot 1 (Top Left)      Index 3 = Dot 4 (Top Right)
Index 1 = Dot 2 (Middle Left)   Index 4 = Dot 5 (Middle Right)
Index 2 = Dot 3 (Bottom Left)   Index 5 = Dot 6 (Bottom Right)
```

Example: `c` = dots 1,4 = `100100`

---

## 6. ESP32 Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | MJPEG stream |
| `/reading` | GET | Returns button state (`0` or `1`) |
| `/braille` | POST | Receives Braille payload, drives motors |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Ollama timeout | Increase `REQUEST_TIMEOUT` in `.env` (default 60s) |
| Tesseract not found | Run `brew install tesseract` |
| `504 Timeout` (Azure) | Check Azure resource region or increase `REQUEST_TIMEOUT` |
| ESP32 unreachable | Connect laptop to ESP32 hotspot first |
| `/reading` keeps timing out | ESP32 is busy playing motors — it will recover automatically |
