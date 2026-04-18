"""
camera_bridge.py

Grabs one frame from the ESP32-CAM MJPEG stream, saves it locally,
and forwards it to the Tactus FastAPI backend for CV processing.

Run:
    python camera_bridge.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERA_URL = "http://192.168.4.1/capture"
BACKEND_URL = "http://127.0.0.1:8000/process-image"
MODE = "read"  # "read" or "describe"

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


def capture_frame_requests(camera_url: str, timeout: int = 10) -> bytes:
    """
    Fallback: fetch a single JPEG directly from the /capture endpoint.
    Returns raw JPEG bytes.
    """
    log("OpenCV unavailable — falling back to requests GET...")
    log(f"Connecting to {camera_url}...")

    response = requests.get(camera_url, timeout=timeout)
    response.raise_for_status()

    image_bytes = response.content
    if not image_bytes:
        raise RuntimeError("Camera returned an empty response.")

    log(f"Frame received via requests ({len(image_bytes):,} bytes)")
    return image_bytes


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
# Output printer
# ---------------------------------------------------------------------------


def print_result(result: dict) -> None:
    log("Parsed JSON response:")
    print()
    print(json.dumps(result, indent=2))
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

    text = result.get("text", "").strip()
    if text:
        print("OCR TEXT:")
        print(f"  {text}")
        print()

    lines = result.get("lines", [])
    if lines:
        print("LINES:")
        for i, line in enumerate(lines, 1):
            print(f"  {i}. {line}")
        print()

    caption = result.get("caption", "").strip()
    if caption:
        print("CAPTION:")
        print(f"  {caption}")
        print()

    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if MODE not in ("read", "describe"):
        print(f"ERROR: Invalid MODE '{MODE}'. Must be 'read' or 'describe'.")
        return

    print()
    print("=" * 50)
    print("  TACTUS — Camera Bridge")
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

    print("\nDone.")


if __name__ == "__main__":
    main()
