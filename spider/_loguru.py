"""Small fallback for environments without the optional loguru package."""

from __future__ import annotations

import logging
from typing import Any


class _FallbackLogger:
    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger("spider")

    def _format(self, message: Any, *args: Any, **kwargs: Any) -> str:
        text = str(message)
        if args or kwargs:
            try:
                text = text.format(*args, **kwargs)
            except Exception:
                pass
        return text

    def debug(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(self._format(message, *args, **kwargs))

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.info(self._format(message, *args, **kwargs))

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(self._format(message, *args, **kwargs))

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self._logger.error(self._format(message, *args, **kwargs))


class _LoguruCompat:
    logger = _FallbackLogger()


loguru = _LoguruCompat()
