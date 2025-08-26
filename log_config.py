import logging


class LogColors:
    BLUE = "\033[94m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter for adding color to log messages.

    This class provides functionality to format log records with color codes
    based on the severity level (INFO, WARNING, ERROR). Each log level is
    assigned a specific color, making log outputs more readable and
    aesthetically pleasing.

    :ivar default_format: The default format string used by the formatter.
    :type default_format: str
    :ivar color_map: A dictionary mapping log levels to their respective color codes.
    :type color_map: dict
    """

    def format(self, record: logging.LogRecord) -> str:
        color_map = {
            logging.INFO: LogColors.BLUE,
            logging.WARNING: LogColors.YELLOW,
            logging.ERROR: LogColors.RED,
        }
        color = color_map.get(record.levelno, "")
        # Inject fields used by the format string
        record.color_on = color
        record.color_off = LogColors.RESET
        return super().format(record)


def setup_logging(for_notebook: bool = False, level: int = logging.INFO) -> None:
    """
    Sets up the logging configuration for an application.

    This function configures the root logger by removing existing handlers, setting
    the log level, and adding a new stream handler with an appropriate formatter.
    The formatter can be adjusted for notebook environments.

    :param for_notebook: A flag indicating if the logging setup is for a notebook
        environment (True) or not (False).
    :param level: The logging level to be set for the root logger (e.g.,
        logging.INFO, logging.DEBUG).
    :return: None
    """
    root_logger = logging.getLogger()

    # Clear existing handlers to prevent duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    if for_notebook:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    else:
        formatter = ColoredFormatter(
            "%(color_on)s%(asctime)s - %(levelname)s - %(message)s%(color_off)s"
        )
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
