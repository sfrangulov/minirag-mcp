"""Who may see the dashboard: one-time sign-in codes, and the sessions they open.

A code is handed only to a holder of the bearer token and is spent by its first use; the
session it opens is a random id carried in a cookie, never the token. Both live in this
process's memory alone, so a restart signs every browser out.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from collections.abc import Callable

COOKIE = "minirag_session"
COOKIE_PATH = "/dashboard"
CODE_TTL = 60.0
SESSION_TTL = 43200.0


class Sessions:
    # Wall-clock time, not monotonic: macOS's monotonic clock stops while the machine
    # sleeps, and a leaked cookie must age through the night all the same.
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        # Each live secret, with the time it was issued.
        self._codes: dict[bytes, float] = {}
        self._sessions: dict[bytes, float] = {}

    def new_code(self) -> str:
        code = secrets.token_urlsafe(32)
        with self._lock:
            now = self._clock()
            _expire(self._codes, now - CODE_TTL)
            self._codes[code.encode("ascii")] = now
        return code

    def redeem(self, code: str) -> str | None:
        """The id of a new session, for a code that is spent by it; None for a code that is
        unknown, spent or expired."""
        with self._lock:
            now = self._clock()
            _expire(self._codes, now - CODE_TTL)
            found = _find(self._codes, code)
            if found is None:
                return None
            del self._codes[found]
            session_id = secrets.token_urlsafe(32)
            _expire(self._sessions, now - SESSION_TTL)
            self._sessions[session_id.encode("ascii")] = now
        return session_id

    def valid(self, session_id: str | None) -> bool:
        with self._lock:
            _expire(self._sessions, self._clock() - SESSION_TTL)
            return _find(self._sessions, session_id) is not None


def _expire(issued: dict[bytes, float], at_or_before: float) -> None:
    for key in [key for key, when in issued.items() if when <= at_or_before]:
        del issued[key]


def _find(issued: dict[bytes, float], sent: str | None) -> bytes | None:
    """The secret of `issued` that `sent` is, compared with each in constant time."""
    # Every secret is ASCII, so no other value can be one.
    if sent is None or not sent.isascii():
        return None
    candidate = sent.encode("ascii")
    found = None
    for key in issued:
        if hmac.compare_digest(candidate, key):
            found = key
    return found
