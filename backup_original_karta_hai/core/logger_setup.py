import logging
from logging.handlers import RotatingFileHandler
from core import config as cfg

LOG_FILE = cfg.LOG_DIR / "app.log"


def get_logger(name):
    root = logging.getLogger("careerpath")
    if not root.handlers:
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler = RotatingFileHandler(LOG_FILE, maxBytes=500000, backupCount=2,
                                           encoding="utf-8")
        file_handler.setFormatter(fmt)
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(file_handler)
        root.addHandler(console)
        root.propagate = False
    return root.getChild(name)


def tail_log(n=60):
    if not LOG_FILE.exists():
        return "No logs yet."
    lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:])