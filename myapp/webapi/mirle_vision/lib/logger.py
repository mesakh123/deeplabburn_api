import logging


class Logger:

    @staticmethod
    def build(name: str, path_to_log_file: str) -> 'Logger':
        file_handler = logging.FileHandler(path_to_log_file)
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.level = logging.INFO
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return Logger(logger, enabled=True)

    def __init__(self, logger: logging.Logger, enabled: bool):
        super().__init__()
        self._logger = logger
        self._enabled = enabled

    def enable(self, enabled: bool):
        self._enabled = enabled

    def log(self, level: int, message: str):
        if self._enabled:
            self._logger.log(level, message)

    def exception(self, message: str):
        if self._enabled:
            self._logger.exception(message)

    def d(self, message: str):
        self.log(logging.DEBUG, message)

    def i(self, message: str):
        self.log(logging.INFO, message)

    def w(self, message: str):
        self.log(logging.WARNING, message)

    def e(self, message: str):
        self.log(logging.ERROR, message)

    def ex(self, message: str):
        self.exception(message)
