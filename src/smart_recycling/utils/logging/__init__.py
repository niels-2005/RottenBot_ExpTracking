import logging

_logger = None


def get_configured_logger() -> logging.Logger:
    """Get the configured logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    global _logger
    if _logger is None:
        _logger = logging.getLogger("my_app")
        _logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s() - %(message)s"
        )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        fh = logging.FileHandler("app.log", mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        _logger.addHandler(ch)
        _logger.addHandler(fh)
    return _logger
