"""
camera_bridge.py

Grabs one frame from the ESP32-CAM MJPEG stream, saves it locally,
and forwards it to the Tactus FastAPI backend for CV processing.

Run:
    python camera_bridge.py
"""

import time
from datetime import datetime
from pathlib import Path

import cv2
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERA_URL   = "http://192.168.4.1/"
BRAILLE_URL  = "http://192.168.4.1/braille"
READING_URL  = "http://192.168.4.1/reading"
BACKEND_URL  = "http://127.0.0.1:8000/process-image"
MODE         = "describe"  # "read" or "describe"
POLL_INTERVAL = 1  # seconds between button polls

CAPTURED_FRAMES_DIR = Path("captured_frames")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_step = 0


def log(msg: str) -> None:
    global _step
    _step += 1
    print(f"[{_step}] {msg}")


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------


def capture_frame_opencv(
    camera_url: str, max_attempts: int = 20, delay: float = 0.5
) -> bytes:
    """
    Try to grab a single frame from the MJPEG stream using OpenCV.
    Retries up to max_attempts times.
    Returns raw JPEG bytes.
    """
    log("Connecting to camera stream via OpenCV...")
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("OpenCV could not open the camera stream.")

    log("Waiting for frame from ESP32-CAM...")
    for attempt in range(1, max_attempts + 1):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            cap.release()
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError("OpenCV failed to encode frame as JPEG.")
            image_bytes = buffer.tobytes()
            log(f"Frame received via OpenCV (attempt {attempt}, {len(image_bytes):,} bytes)")
            return image_bytes

        print(f"    ... attempt {attempt}/{max_attempts}, no frame yet — retrying in {delay}s")
        time.sleep(delay)

    cap.release()
    raise RuntimeError(f"No valid frame received after {max_attempts} attempts.")


def capture_frame_requests(camera_url: str, timeout: int = 30) -> bytes:
    """
    Fallback: parse a single JPEG frame out of the MJPEG stream.
    Scans the byte stream for JPEG start (\\xff\\xd8) and end (\\xff\\xd9) markers.
    Returns raw JPEG bytes.
    """
    log("OpenCV unavailable — falling back to requests-based MJPEG parser...")
    log(f"Connecting to {camera_url} (streaming mode)...")

    response = requests.get(camera_url, stream=True, timeout=timeout)
    response.raise_for_status()

    log("Waiting for JPEG frame in stream...")

    buffer = b""
    start_marker = b"\xff\xd8"
    end_marker = b"\xff\xd9"

    for chunk in response.iter_content(chunk_size=1024):
        if not chunk:
            continue
        buffer += chunk

        start = buffer.find(start_marker)
        end = buffer.find(end_marker)

        if start != -1 and end != -1 and end > start:
            image_bytes = buffer[start : end + 2]
            log(f"Frame received via requests parser ({len(image_bytes):,} bytes)")
            return image_bytes

    raise RuntimeError("Stream ended without a complete JPEG frame.")


# ---------------------------------------------------------------------------
# Debug image save
# ---------------------------------------------------------------------------


def save_debug_image(image_bytes: bytes) -> Path:
    """Save JPEG bytes to captured_frames/ with a timestamped filename."""
    CAPTURED_FRAMES_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = CAPTURED_FRAMES_DIR / f"frame_{timestamp}.jpg"
    filepath.write_bytes(image_bytes)
    log(f"Saved debug image: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Backend upload
# ---------------------------------------------------------------------------


def upload_to_backend(image_bytes: bytes, backend_url: str, mode: str) -> dict:
    """Upload the JPEG frame to the FastAPI backend and return parsed JSON."""
    log(f"Uploading image to FastAPI backend ({mode} mode)...")

    response = requests.post(
        backend_url,
        files={"imageFile": ("frame.jpg", image_bytes, "image/jpeg")},
        data={"mode": mode},
        timeout=30,
    )

    log(f"FastAPI responded with HTTP {response.status_code}")

    try:
        return response.json()
    except Exception:
        raise RuntimeError(
            f"Backend returned non-JSON response: {response.text[:300]}"
        )


# ---------------------------------------------------------------------------
# ESP32 Braille sender
# ---------------------------------------------------------------------------


def send_to_esp32(payload: str, braille_url: str) -> None:
    """POST the comma-separated Braille payload back to the ESP32-CAM."""
    log(f"Sending payload back to ESP32-CAM at {braille_url}...")
    try:
        response = requests.post(
            braille_url,
            data=payload,
            headers={"Content-Type": "text/plain"},
            timeout=5,
        )
        log(f"ESP32 received it! Response: {response.text.strip()}")
    except requests.exceptions.Timeout:
        print("    Warning: ESP32 did not respond in time (timeout after 5s).")
    except requests.exceptions.ConnectionError:
        print("    Warning: Could not reach ESP32 — may have dropped off the network.")
    except Exception as e:
        print(f"    Warning: ESP32 send failed — {e}")


# ---------------------------------------------------------------------------
# Output printer
# ---------------------------------------------------------------------------


def print_result(result: dict) -> None:
    log("Parsed JSON response:")
    print()

    if not result.get("success"):
        print("=" * 50)
        print("ERROR:", result.get("error", "Unknown error"))
        print("=" * 50)
        return

    print("=" * 50)

    if result.get("guidance"):
        print("GUIDANCE:")
        print(f"  {result['guidance']}")
        print()

    caption = result.get("caption", "").strip()
    if caption:
        print("CAPTION:")
        print(f"  {caption}")
        print()

    text = result.get("text", "").strip()
    lines = result.get("lines", [])
    if lines:
        print("OCR TEXT:")
        for i, line in enumerate(lines, 1):
            print(f"  {i}. {line}")
        print()
    elif text:
        print("OCR TEXT:")
        print(f"  {text}")
        print()

    braille_debug = result.get("braille_debug", [])
    braille_payload = result.get("braille_payload", "")
    if braille_debug:
        print("BRAILLE TRANSLATION:")
        for entry in braille_debug:
            print(f"  {entry}")
        print()
    if braille_payload:
        print("PAYLOAD TO ESP32:")
        print(f"  {braille_payload}")
        print()

    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Capture one frame, process it, and send braille to ESP32."""
    global _step
    _step = 0

    print()
    print("=" * 50)
    print("  TACTUS — Pipeline triggered by button press")
    print(f"  Camera : {CAMERA_URL}")
    print(f"  Backend: {BACKEND_URL}")
    print(f"  Mode   : {MODE}")
    print("=" * 50)
    print()

    # Step 1-3: Capture frame
    try:
        image_bytes = capture_frame_opencv(CAMERA_URL)
    except Exception as opencv_err:
        print(f"    OpenCV failed: {opencv_err}")
        print("    Switching to requests fallback...")
        try:
            image_bytes = capture_frame_requests(CAMERA_URL)
        except Exception as req_err:
            print(f"\nFATAL: Could not capture frame from camera.")
            print(f"  OpenCV error  : {opencv_err}")
            print(f"  Requests error: {req_err}")
            return

    # Step 4: Save debug copy
    try:
        save_debug_image(image_bytes)
    except Exception as e:
        print(f"    Warning: could not save debug image — {e}")

    # Step 5-7: Upload and print
    try:
        result = upload_to_backend(image_bytes, BACKEND_URL, MODE)
        print_result(result)
    except Exception as e:
        print(f"\nFATAL: Backend upload failed — {e}")
        return

    # Step 8-9: Forward Braille payload to ESP32 (cap at 20 cells for playback)
    payload = result.get("braille_payload", "")
    if payload:
        cells = payload.split(",")
        truncated = ",".join(cells[:20])
        cell_count = len(cells[:20])
        send_to_esp32(truncated, BRAILLE_URL)
        wait_time = cell_count * 2.0
        log(f"Waiting {wait_time:.0f}s for ESP32 to finish playing {cell_count} cells...")
        time.sleep(wait_time)
    else:
        print("    (No Braille payload to send.)")

    print("\nPipeline complete. Resuming polling...\n")


def main() -> None:
    if MODE not in ("read", "describe"):
        print(f"ERROR: Invalid MODE '{MODE}'. Must be 'read' or 'describe'.")
        return

    print()
    print("=" * 50)
    print("  TACTUS — Waiting for button press")
    print(f"  Polling: {READING_URL} every {POLL_INTERVAL}s")
    print(f"  Mode   : {MODE}")
    print("=" * 50)
    print()

    while True:
        try:
            response = requests.get(READING_URL, stream=True, timeout=5)
            value = next(response.iter_content(chunk_size=32), b"0").decode().strip()
            response.close()
            time.sleep(0.5)

            if value == "1":
                print("Button pressed! Starting pipeline...")
                run_pipeline()
            else:
                print(f"  Waiting... (button={value})", end="\r")

        except requests.exceptions.Timeout:
            print("  Warning: /reading timed out, retrying...", end="\r")
        except requests.exceptions.ConnectionError:
            print("  Warning: ESP32 unreachable, retrying...", end="\r")
        except Exception as e:
            print(f"  Warning: poll error — {e}", end="\r")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
