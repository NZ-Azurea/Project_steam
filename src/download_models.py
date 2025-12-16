import os
import logging
import requests
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

URL = (
    "https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF/"
    "resolve/main/openai_gpt-oss-20b-MXFP4.gguf?download=true"
)

MODELS_DIR = "models"
MODEL_NAME = "openai_gpt-oss-20b-MXFP4.gguf"
DST_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

os.makedirs(MODELS_DIR, exist_ok=True)

headers = {}
downloaded = 0

if os.path.exists(DST_PATH):
    downloaded = os.path.getsize(DST_PATH)
    headers["Range"] = f"bytes={downloaded}-"
    logger.info("Resuming download at %.2f GB", downloaded / (1024**3))

logger.info("Starting model download")

with requests.get(URL, stream=True, headers=headers, timeout=60) as r:
    r.raise_for_status()

    total = r.headers.get("Content-Length")
    total = int(total) + downloaded if total else None

    mode = "ab" if downloaded > 0 else "wb"

    bar_width = 40
    last_log_time = 0.0
    log_interval = 1.0  # seconds

    with open(DST_PATH, mode) as f:
        current = downloaded
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue

            f.write(chunk)
            current += len(chunk)

            now = time.time()
            if total and now - last_log_time >= log_interval:
                percent = current / total
                filled = int(bar_width * percent)
                bar = "#" * filled + "-" * (bar_width - filled)

                sys.stdout.write(
                    f"\r[DL] [{bar}] {percent*100:6.2f}% "
                    f"{current/(1024**3):.2f}/{total/(1024**3):.2f} GB"
                )
                sys.stdout.flush()
                last_log_time = now

sys.stdout.write("\n")
logger.info("Model downloaded successfully: %s", DST_PATH)
