"""Typed event bus for pyflexaudio."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

__all__ = ["EventBus"]

logger = logging.getLogger("pyflexaudio.events")


class EventBus:
    """Typed event bus.

    Handlers are registered per event type and called inline on the emitting
    thread. Thread-safe via a short-held lock; handler execution happens
    outside the lock to avoid deadlocks.
    """

    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable]] = {}
        self._lock = threading.Lock()

    def on(self, event_type: type, handler: Callable) -> None:
        """Register a handler for the given event type."""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def off(self, event_type: type, handler: Callable) -> None:
        """Unregister a handler for the given event type."""
        with self._lock:
            if event_type in self._handlers:
                try:
                    self._handlers[event_type].remove(handler)
                except ValueError:
                    pass

    def emit(self, event: object) -> None:
        """Emit an event, calling all registered handlers inline.

        A snapshot of the handler list is taken under the lock so the lock is
        held only for the duration of the list copy, not during handler
        execution. Each handler runs independently; an exception in one handler
        does not prevent the remaining handlers from running.
        """
        with self._lock:
            handlers = list(self._handlers.get(type(event), []))

        for handler in handlers:
            start = time.monotonic()
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "EventBus handler error for %s", type(event).__name__
                )
            elapsed = time.monotonic() - start
            if elapsed > 0.1:  # 100 ms
                logger.warning(
                    "EventBus handler %s took %.0fms for %s",
                    handler,
                    elapsed * 1000,
                    type(event).__name__,
                )

    def clear(self) -> None:
        """Unregister all handlers."""
        with self._lock:
            self._handlers.clear()

    def has_handlers(self, event_type: type) -> bool:
        """Return True if at least one handler is registered for event_type."""
        with self._lock:
            return bool(self._handlers.get(event_type))

    def handler_count(self, event_type: type) -> int:
        """Return the number of handlers registered for event_type."""
        with self._lock:
            return len(self._handlers.get(event_type, []))
