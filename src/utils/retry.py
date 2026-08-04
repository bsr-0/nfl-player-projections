"""Retry decorator with exponential backoff for network operations.

Per Agent Directive V7 Section 19: production data pipelines must handle
failures gracefully without corrupting downstream state.
"""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Tuple, Type

logger = logging.getLogger(__name__)

# Default exceptions that indicate transient network failures
_NETWORK_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

try:
    import requests
    _NETWORK_EXCEPTIONS = (
        *_NETWORK_EXCEPTIONS,
        requests.ConnectionError,
        requests.Timeout,
        requests.HTTPError,
    )
except ImportError:
    pass

try:
    from urllib.error import URLError
    _NETWORK_EXCEPTIONS = (*_NETWORK_EXCEPTIONS, URLError)
except ImportError:
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[BaseException], ...] = _NETWORK_EXCEPTIONS,
):
    """Decorator that retries a function on transient failures.

    Uses exponential backoff with jitter to avoid thundering herd.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        exceptions: Tuple of exception types to catch and retry.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_retries:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, max_retries + 1, exc,
                        )
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.25)
                    wait = delay + jitter
                    logger.warning(
                        "%s attempt %d/%d failed (%s), retrying in %.1fs",
                        func.__name__, attempt + 1, max_retries + 1,
                        type(exc).__name__, wait,
                    )
                    time.sleep(wait)
            raise last_exception  # Should not reach here

        return wrapper

    return decorator
