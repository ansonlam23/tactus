import os

from dotenv import load_dotenv

load_dotenv()

VISION_ENDPOINT: str = os.getenv("VISION_ENDPOINT", "").rstrip("/")
VISION_KEY: str = os.getenv("VISION_KEY", "")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
VISION_BACKEND: str = os.getenv("VISION_BACKEND", "azure")  # "azure" or "local"
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "moondream")
