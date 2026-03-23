

import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=log_file,
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    level=logging.INFO
)

logger = logging.getLogger()


if __name__ == "__main__":
    logger.info("Testing info message")
    logger.warning("Testing warning message")
    logger.error("Testing error message")
    print("Logger test completed. Check the logs directory for the log file.")