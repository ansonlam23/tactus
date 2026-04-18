import os

from dotenv import load_dotenv

load_dotenv()

VISION_ENDPOINT: str = os.getenv("VISION_ENDPOINT", "").rstrip("/")
VISION_KEY: str = os.getenv("VISION_KEY", "")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
