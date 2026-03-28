"""Workload generator for the Online Boutique frontend.

Simulates realistic user journeys (browse → view product → add to cart → checkout)
at a configurable RPS with sine, step, or constant load patterns.
"""

import math
import queue
import threading
import time
from collections import deque
from typing import Literal

import click
import requests

# All product IDs present in the default Online Boutique catalogue
PRODUCT_IDS = [
    "OLJCESPC7Z",
    "66VCHSJNUP",
    "1YMWWN1N4O",
    "L9ECAV7KIM",
    "2ZYFJ3GM2N",
    "0PUK6V6EV0",
    "LS4PSXUNUM",
    "9SIQT8TOJO",
    "6E92ZMYYFZ",
]

# Fake checkout payload — realistic enough to pass validation
_CHECKOUT_FORM = {
    "email": "test@example.com",
    "street_address": "1600 Amphitheatre Pkwy",
    "zip_code": "94043",
    "city": "Mountain View",
    "state": "CA",
    "country": "US",
    "credit_card_number": "4432801561520454",
    "credit_card_expiration_month": "1",
    "credit_card_expiration_year": "2030",
    "credit_card_cvv": "672",
}

PatternType = Literal["sine", "step", "constant"]


class WorkloadGenerator:
    """Generates HTTP traffic against the Online Boutique frontend."""

    def __init__(self, frontend_url: str = "http://localhost:8080", quiet: bool = False) -> None:
        self.frontend_url = frontend_url.rstrip("/")
        self._session = requests.Session()
        self._quiet = quiet  # suppresses SLO violation prints (use when run from orchestrator)
        # Rolling window of (completed_at, latency_s) tuples for SLO tracking
        self._latency_window: deque[tuple[float, float]] = deque()
        self._window_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # User journey steps
    # ------------------------------------------------------------------

    def _browse_homepage(self) -> float:
        """GET / — returns latency in seconds."""
        t0 = time.perf_counter()
        self._session.get(self.frontend_url + "/", timeout=10)
        return time.perf_counter() - t0

    def _view_product(self, product_id: str) -> float:
        """GET /product/{id} — returns latency in seconds."""
        t0 = time.perf_counter()
        self._session.get(f"{self.frontend_url}/product/{product_id}", timeout=10)
        return time.perf_counter() - t0

    def _add_to_cart(self, product_id: str) -> float:
        """POST /cart — returns latency in seconds."""
        t0 = time.perf_counter()
        self._session.post(
            self.frontend_url + "/cart",
            data={"product_id": product_id, "quantity": 1},
            timeout=10,
        )
        return time.perf_counter() - t0

    def _checkout(self) -> float:
        """POST /cart/checkout — returns latency in seconds."""
        t0 = time.perf_counter()
        self._session.post(
            self.frontend_url + "/cart/checkout",
            data=_CHECKOUT_FORM,
            timeout=10,
        )
        return time.perf_counter() - t0

    def _user_journey(self) -> None:
        """Execute one full user journey and record latencies."""
        import random

        product_id = random.choice(PRODUCT_IDS)
        for step in (
            self._browse_homepage,
            lambda: self._view_product(product_id),
            lambda: self._add_to_cart(product_id),
            self._checkout,
        ):
            try:
                latency = step()
            except requests.exceptions.RequestException:
                latency = float("nan")
            now = time.time()
            with self._window_lock:
                self._latency_window.append((now, latency))

    # ------------------------------------------------------------------
    # RPS pattern helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rps_at(t: float, base_rps: float, pattern: PatternType) -> float:
        """Return the target RPS at elapsed time *t* (seconds)."""
        if pattern == "sine":
            return base_rps + base_rps * 0.5 * math.sin(2 * math.pi * t / 60)
        if pattern == "step":
            return base_rps * 2 if int(t / 30) % 2 == 1 else base_rps
        return base_rps  # constant

    # ------------------------------------------------------------------
    # SLO monitoring
    # ------------------------------------------------------------------

    def _check_slo(self, window_seconds: float = 10.0, p95_threshold: float = 0.5) -> None:
        """Log a warning if p95 latency over the last *window_seconds* exceeds threshold."""
        now = time.time()
        cutoff = now - window_seconds
        with self._window_lock:
            recent = [lat for ts, lat in self._latency_window if ts >= cutoff and not math.isnan(lat)]
            # Prune entries older than 2x the window to bound memory use
            while self._latency_window and self._latency_window[0][0] < now - window_seconds * 2:
                self._latency_window.popleft()

        if not recent:
            return

        recent.sort()
        p95 = recent[int(len(recent) * 0.95)]
        if p95 > p95_threshold and not self._quiet:
            print(
                f"[loadgen] SLO VIOLATION  p95={p95*1000:.0f}ms > {p95_threshold*1000:.0f}ms"
                f"  (n={len(recent)} samples in last {window_seconds:.0f}s)"
            )

    def current_p95(self, window_seconds: float = 10.0) -> float | None:
        """Return p95 latency (seconds) over the last *window_seconds*, or None if no data."""
        now = time.time()
        cutoff = now - window_seconds
        with self._window_lock:
            recent = [lat for ts, lat in self._latency_window if ts >= cutoff and not math.isnan(lat)]
        if not recent:
            return None
        recent.sort()
        return recent[int(len(recent) * 0.95)]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self, duration_seconds: float, base_rps: float, pattern: PatternType) -> None:
        """Inner loop executed on the background thread."""
        start = time.time()
        next_slo_check = start + 10.0
        # Use a token-bucket approach: track the next allowed request time
        next_req_time = start

        while not self._stop_event.is_set():
            elapsed = time.time() - start
            if elapsed >= duration_seconds:
                break

            now = time.time()
            if now < next_req_time:
                time.sleep(max(0.0, next_req_time - now))

            rps = max(0.1, self._rps_at(elapsed, base_rps, pattern))
            # Fire a journey on a short-lived thread so we don't block the pacing loop
            threading.Thread(target=self._user_journey, daemon=True).start()

            next_req_time += 1.0 / rps

            if time.time() >= next_slo_check:
                self._check_slo()
                next_slo_check += 10.0

        self._stop_event.set()

    def run(
        self,
        duration_seconds: float = 300,
        base_rps: float = 5,
        pattern: PatternType = "sine",
        block: bool = False,
    ) -> threading.Thread:
        """Start the workload generator.

        Args:
            duration_seconds: How long to run before stopping automatically.
            base_rps:         Baseline requests per second.
            pattern:          Load pattern — ``"sine"``, ``"step"``, or ``"constant"``.
            block:            If True, block the caller until the run completes.

        Returns:
            The background ``threading.Thread`` (already started).
        """
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(duration_seconds, base_rps, pattern),
            daemon=True,
            name="loadgen",
        )
        print(
            f"[loadgen] Starting  pattern={pattern}  base_rps={base_rps}"
            f"  duration={duration_seconds}s  target={self.frontend_url}"
        )
        self._thread.start()
        if block:
            self._thread.join()
        return self._thread

    def stop(self) -> None:
        """Signal the generator to stop and wait for the thread to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--url", default="http://localhost:8080", show_default=True, help="Frontend URL.")
@click.option("--duration", default=300, show_default=True, help="Run duration in seconds.")
@click.option("--rps", default=5.0, show_default=True, help="Base requests per second.")
@click.option(
    "--pattern",
    default="sine",
    show_default=True,
    type=click.Choice(["sine", "step", "constant"]),
    help="Load pattern.",
)
def main(url: str, duration: int, rps: float, pattern: str) -> None:
    """Generate HTTP load against the Online Boutique frontend."""
    gen = WorkloadGenerator(frontend_url=url)
    gen.run(duration_seconds=duration, base_rps=rps, pattern=pattern, block=True)
    print("[loadgen] Done.")


if __name__ == "__main__":
    main()
