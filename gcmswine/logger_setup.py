# logger_setup.py
import logging

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