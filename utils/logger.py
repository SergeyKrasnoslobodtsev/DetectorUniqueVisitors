import logging
import logging.config
import sys
import io

from colorlog import ColoredFormatter
from functools import wraps


def capture_output(func):
    """Wrapper to capture print output."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
    return wrapper


class LoggerFactory:
    @staticmethod
    def configure():
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )

        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'colored': {
                    '()': ColoredFormatter,
                    'format': '%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s\n',
                    'log_colors': {
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white',
                    },
                },
            },
            'handlers': {
                'default': {
                    'level': 'DEBUG',
                    'formatter': 'colored',
                    'class': 'logging.StreamHandler',
                },
            },
            'loggers': {
                '': {  # root logger
                    'handlers': ['default'],
                    'level': 'DEBUG',
                    'propagate': True
                },
            }
        })

    @staticmethod
    def get_logger(name):
        return logging.getLogger(name)


if __name__ == "__main__":
    LoggerFactory.configure()
    logger = LoggerFactory.get_logger('nm')
    logger.debug("Some debugging output")
    logger.info("Some info output")
    logger.error("Some error output")
    logger.warning("Some warning output")
