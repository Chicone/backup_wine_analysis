# logger_setup.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # <--- Add this line
    handlers=[
        logging.FileHandler("log_buffer.txt", mode="a"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("wine_analysis")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("log_buffer.txt")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
file_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(file_handler)

# Optional: also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def logger_raw(message: str):
    """Append raw message without timestamp to the log buffer file."""
    try:
        with open("log_buffer.txt", "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        logger.error(f"Failed to write raw log: {e}")