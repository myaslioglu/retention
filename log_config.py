import logging


class LogColors:
    BLUE = '\033[94m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color_map = {
            logging.INFO: LogColors.BLUE,
            logging.WARNING: LogColors.YELLOW,
            logging.ERROR: LogColors.RED,
        }
        color = color_map.get(record.levelno, '')
        # Inject fields used by the format string
        record.color_on = color
        record.color_off = LogColors.RESET
        return super().format(record)


def setup_logging(for_notebook: bool = False, level: int = logging.INFO) -> None:
    """
    Configure root logging so logs appear in both CLI and Jupyter.

    - Clears existing handlers to avoid duplicates when reconfiguring
    - Uses colorized output in terminal; plain output in notebooks
    """
    root_logger = logging.getLogger()

    # Clear existing handlers to prevent duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    if for_notebook:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    else:
        formatter = ColoredFormatter('%(color_on)s%(asctime)s - %(levelname)s - %(message)s%(color_off)s')
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)


