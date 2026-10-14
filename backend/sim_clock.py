"""
Simulation Clock for the AI Elevator System.

Separates simulated time from wall-clock time while maintaining authoritative
calendar time in Asia/Kolkata timezone.

Features:
  1. Authoritative simulation calendar time (date, time, weekday) in Asia/Kolkata.
  2. Running mode: advances simulated seconds using monotonic wall-clock time
     scaled by configurable speed (1x, 2x, 5x, etc.).
  3. Pause / Resume semantics: freeze progression on pause, resume without jumps.
  4. Speed changes smoothly re-anchor without loss or duplicate elapsed time.
  5. Deterministic mode for offline unit testing without real-time sleep.
  6. Derives hour, minute, day-of-week features directly for LSTM inference.
  7. Thread-safe: all state mutations and queries are guarded by a threading.Lock.
"""

from datetime import datetime, date, time as dt_time, timedelta
import threading
import time as _realtime
from typing import Any, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore


DEFAULT_TIMEZONE = "Asia/Kolkata"


class SimClock:
    """Authoritative, thread-safe simulation clock with calendar progression."""

    def __init__(
        self,
        *,
        deterministic: bool = False,
        timezone: str = DEFAULT_TIMEZONE,
        initial_date: str | None = None,
        initial_time: str | None = None,
    ):
        self._lock = threading.Lock()
        self._deterministic = bool(deterministic)
        try:
            self._tz = ZoneInfo(timezone)
        except Exception:
            self._tz = ZoneInfo("UTC")

        self._mode: str = "custom"           # "custom" | "real_time"
        self._speed: float = 1.0             # 1.0, 2.0, 5.0
        self._is_running: bool = False
        self._tick_delta: float = 1.0        # simulated seconds per deterministic tick
        self._sim_time: float = 0.0          # accumulated simulated seconds elapsed
        self._last_wall_mark: float | None = None

        # Authoritative base calendar datetime (Asia/Kolkata)
        self._calendar_base = self._parse_datetime(initial_date, initial_time)

    # ── internal time synchronization ────────────────────────────────────────

    def _parse_datetime(self, date_str: str | None, time_str: str | None) -> datetime:
        """Parses date and time strings into a timezone-aware datetime."""
        now_in_tz = datetime.now(self._tz)
        target_date = now_in_tz.date()

        if date_str:
            clean_date = date_str.strip()
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
                try:
                    target_date = datetime.strptime(clean_date, fmt).date()
                    break
                except ValueError:
                    pass

        target_hour = 9
        target_min = 0
        target_sec = 0

        if time_str:
            clean_time = time_str.strip()
            # Try 12-hour formats e.g. "09:00:00 AM", "9:00 AM"
            for fmt in ("%I:%M:%S %p", "%I:%M %p", "%I:%M:%S%p", "%I:%M%p",
                        "%H:%M:%S", "%H:%M"):
                try:
                    parsed_t = datetime.strptime(clean_time, fmt).time()
                    target_hour = parsed_t.hour
                    target_min = parsed_t.minute
                    target_sec = parsed_t.second
                    break
                except ValueError:
                    pass

        base = datetime.combine(
            target_date,
            dt_time(target_hour, target_min, target_sec),
            tzinfo=self._tz
        )
        return base

    def _update_live_elapsed_locked(self) -> None:
        """Updates accumulated simulated seconds from wall clock if running in live mode."""
        if not self._deterministic and self._is_running and self._last_wall_mark is not None:
            now_wall = _realtime.monotonic()
            dt_wall = max(0.0, now_wall - self._last_wall_mark)
            self._sim_time += dt_wall * self._speed
            self._last_wall_mark = now_wall

    # ── queries ──────────────────────────────────────────────────────────────

    def now(self) -> float:
        """Current elapsed simulated seconds since clock start/reset."""
        with self._lock:
            self._update_live_elapsed_locked()
            return self._sim_time

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def tick_delta(self) -> float:
        return self._tick_delta

    @property
    def timezone_name(self) -> str:
        return str(self._tz)

    def get_calendar_datetime(self) -> datetime:
        """
        Returns authoritative current simulated datetime in Asia/Kolkata.
        In custom mode: calendar_base + accumulated simulated seconds.
        In real_time mode: active wall-clock time in Asia/Kolkata.
        """
        with self._lock:
            self._update_live_elapsed_locked()
            if self._mode == "real_time" and not self._deterministic:
                return datetime.now(self._tz)
            return self._calendar_base + timedelta(seconds=self._sim_time)

    def get_time_features(self) -> Tuple[int, int, int, str]:
        """
        Derives time features matching LSTM schema:
        Returns: (hour [0-23], minute [0-59], day_of_week [0=Mon..6=Sun], weekday_name)
        """
        dt = self.get_calendar_datetime()
        return (dt.hour, dt.minute, dt.weekday(), dt.strftime("%A"))

    def get_status_dict(self) -> dict[str, Any]:
        """Returns comprehensive status for API responses and dashboard telemetry."""
        dt = self.get_calendar_datetime()
        with self._lock:
            return {
                "sim_time_seconds": round(self._sim_time, 2),
                "calendar_datetime_iso": dt.isoformat(),
                "calendar_date_str": dt.strftime("%Y-%m-%d"),
                "calendar_time_str": dt.strftime("%I:%M:%S %p"),
                "calendar_time_24h": dt.strftime("%H:%M:%S"),
                "weekday": dt.strftime("%A"),
                "weekday_idx": dt.weekday(),
                "mode": self._mode,
                "speed": self._speed,
                "is_running": self._is_running,
                "timezone": str(self._tz),
            }

    # ── lifecycle & control ──────────────────────────────────────────────────

    def start(self) -> None:
        """Starts time progression."""
        with self._lock:
            if not self._is_running:
                self._is_running = True
                self._last_wall_mark = _realtime.monotonic()

    def pause(self) -> None:
        """Freezes simulated time without clearing accumulated elapsed time or base."""
        with self._lock:
            if self._is_running:
                self._update_live_elapsed_locked()
                self._is_running = False
                self._last_wall_mark = None

    def resume(self) -> None:
        """Continues simulated time progression from exact paused moment."""
        with self._lock:
            if not self._is_running:
                self._is_running = True
                self._last_wall_mark = _realtime.monotonic()

    def set_speed(self, speed: float) -> float:
        """
        Changes simulation speed scaling factor (1x, 2x, 5x, etc.).
        Re-anchors elapsed time up to this instant to avoid time jumps.
        """
        with self._lock:
            self._update_live_elapsed_locked()
            self._speed = max(0.1, float(speed))
            if self._is_running:
                self._last_wall_mark = _realtime.monotonic()
            return self._speed

    def set_mode(self, mode: str) -> str:
        """Switch between 'custom' and 'real_time' modes."""
        with self._lock:
            m = str(mode).strip().lower()
            if m in ("real_time", "realtime", "live"):
                self._mode = "real_time"
            else:
                self._mode = "custom"
            return self._mode

    def set_custom_time(
        self,
        date_str: str | None = None,
        time_str: str | None = None,
        dt: datetime | None = None,
    ) -> datetime:
        """
        Sets a new authoritative simulation calendar date/time.
        Resets elapsed simulated seconds to 0.0 for this new anchor.
        Does NOT clear elevators or passengers.
        """
        with self._lock:
            if dt is not None:
                if dt.tzinfo is None:
                    self._calendar_base = dt.replace(tzinfo=self._tz)
                else:
                    self._calendar_base = dt.astimezone(self._tz)
            else:
                self._calendar_base = self._parse_datetime(date_str, time_str)

            self._sim_time = 0.0
            self._mode = "custom"
            if self._is_running:
                self._last_wall_mark = _realtime.monotonic()
            return self._calendar_base

    def set_time(self, hour: int = 9, minute: int = 0, second: int = 0) -> datetime:
        """Convenience method to set clock time using hour, minute, second."""
        time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
        return self.set_custom_time(time_str=time_str)

    def advance(self, dt: float | None = None) -> float:
        """
        Advances the simulation clock by *dt* simulated seconds.
        In deterministic mode, increments _sim_time directly.
        In live mode, updates from wall clock first, then adds dt.
        """
        if dt is None:
            dt = self._tick_delta
        with self._lock:
            if not self._deterministic and self._is_running:
                self._update_live_elapsed_locked()
            self._sim_time += float(dt)
            return self._sim_time

    def reset(self) -> None:
        """Resets elapsed simulated seconds to zero and stops running."""
        with self._lock:
            self._sim_time = 0.0
            self._is_running = False
            self._last_wall_mark = None

    def set_tick_delta(self, delta: float) -> None:
        with self._lock:
            self._tick_delta = max(0.001, float(delta))

    # ── wall-clock helpers (used by batch window timing) ─────────────────────

    def wall_elapsed_since(self, wall_mark: float) -> float:
        """Seconds elapsed on the *wall* clock since `wall_mark`."""
        return _realtime.monotonic() - wall_mark

    @staticmethod
    def wall_now() -> float:
        """Current wall-clock monotonic timestamp."""
        return _realtime.monotonic()

    # ── repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        dt = self.get_calendar_datetime()
        mode = "deterministic" if self._deterministic else "live"
        status = "running" if self._is_running else "paused"
        return (
            f"SimClock({dt.strftime('%Y-%m-%d %H:%M:%S')} {str(self._tz)}, "
            f"t={self._sim_time:.2f}s, speed={self._speed}x, {status}, {mode})"
        )
