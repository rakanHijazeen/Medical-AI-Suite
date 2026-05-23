import logging
from pathlib import Path

def setup_logger(name="medical_ai"):
    """
    Sets up a dual-destination context-aware logger for the application.
    Clutters your terminal with minimal, clean text while appending exact,
    auditable metrics to your app.log file.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if initialized multiple times across scripts
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)

    # 1. Resolve workspace paths dynamically relative to this file
    # This places the 'logs' folder smoothly at your root project level
    root_dir = Path(__file__).resolve().parents[1]
    log_dir = root_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"

    # 2. Define target-specific format layouts
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_formatter = logging.Formatter(
        "⚡ [%(levelname)s] %(message)s"
    )

    # 3. File Handler: Captures technical fingerprints, paths, and raw timestamps
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # 4. Console Handler: Stripped down and highly legible for local development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # 5. Lock down handles onto the unified logger instance
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Single instance initialization for clean global imports across components
logger = setup_logger()