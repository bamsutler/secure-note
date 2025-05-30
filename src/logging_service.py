import logging
from rich.logging import RichHandler # Using Rich for nice console output

class LoggingService:
    _loggers = {}

    @staticmethod
    def get_logger(name: str, level: int = logging.INFO):
        """
        Retrieves a logger instance. If it doesn't exist, it's created and configured.
        Args:
            name (str): The name for the logger (e.g., __name__ of the calling module).
            level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        Returns:
            logging.Logger: The configured logger instance.
        """
        if name in LoggingService._loggers:
            return LoggingService._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Cache the logger
        LoggingService._loggers[name] = logger
        return logger

    @staticmethod
    def setup_root_logger(level: int = logging.INFO, handler=None):
        """
        Configures the root logger. This is useful for setting a baseline logging behavior.
        Args:
            level (int): The logging level for the root logger.
            handler (logging.Handler, optional): A specific handler to add to the root logger.
                                                If None, a RichHandler is used.
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear existing handlers from root to avoid duplication if this is called multiple times
        # or if other libraries have added handlers.
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
            h.close() # Close the handler before removing

        if handler is None:
            rich_handler = RichHandler(rich_tracebacks=True, markup=True)
            rich_handler.setFormatter(logging.Formatter("%(levelname)s: %(name)s: %(message)s", datefmt="[%X]"))
            root_logger.addHandler(rich_handler)
        else:
            root_logger.addHandler(handler)
        
        logging.info(f"Root logger configured with level {logging.getLevelName(level)}.")

