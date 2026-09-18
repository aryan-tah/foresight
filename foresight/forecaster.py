"""TimesFM 3.0 wrapper: forecast a Series and honestly score how far to trust it.

Every forecast comes with a backtest: the last `horizon` points are hidden, the
model forecasts them, and the error is compared against a seasonal-naive
baseline ("next week looks like last week"). That gives the agent a real number
to ground its confidence in, instead of vibes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import numpy as np

from .series import Series, sig

MIN_HISTORY = 30
MAX_HORIZON = 180
WEEK = 7
LONG_HORIZON = 45  # from here on, charts show two years of history
# Deciles 0.1..0.9 come back in this order; pick the ones the fan chart uses.
_Q = {"p10": 0, "p30": 2, "p70": 6, "p90": 8}


class Model(Protocol):
    def predict(
        self, values: np.ndarray, horizon: int, positive: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (median forecast [h], deciles [h, 9])."""


class TimesFM:
    """Lazy, thread-safe handle on the TimesFM 3.0 checkpoint (CPU is plenty)."""

    checkpoint = "google/timesfm-3.0-pytorch"

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                from timesfm3 import TimesFM3Forecaster

                self._model = TimesFM3Forecaster.from_pretrained(
                    self.checkpoint, device="cpu"
                )

    def predict(self, values, horizon, positive):
        self.load()
        with self._lock:
            out = self._model.predict(
                np.asarray(values, dtype=np.float32),
                horizon=horizon,
                return_quantiles=True,
                make_positive=positive,
            )
        return out.forecast, out.quantiles


@dataclass
class Forecast:
    series: Series
    dates: list[date]
    point: list[float]
    bands: dict[str, list[float]]  # p10 / p30 / p70 / p90
    backtest: dict | None

    def summary(self) -> dict:
        """What the LLM sees: enough to reason with, not the full arrays."""
        s, h = self.series, len(self.point)
        recent = float(np.mean(s.values[-h:]))
        ahead = float(np.mean(self.point))
        step = max(1, h // 6)
        peak = int(np.argmax(self.point))
        return {
            "series_id": s.id,
            "label": s.label,
            "unit": s.unit,
            "horizon_days": h,
            "last_observed": {"date": s.dates[-1].isoformat(), "value": _r(s.values[-1])},
            "forecast_end": {
                "date": self.dates[-1].isoformat(),
                "median": _r(self.point[-1]),
                "p10": _r(self.bands["p10"][-1]),
                "p90": _r(self.bands["p90"][-1]),
            },
            "forecast_peak": {
                "date": self.dates[peak].isoformat(),
                "weekday": self.dates[peak].strftime("%A"),
                "median": _r(self.point[peak]),
                "p10": _r(self.bands["p10"][peak]),
                "p90": _r(self.bands["p90"][peak]),
            },
            "mean_next_period": _r(ahead),
            "mean_same_length_recent_period": _r(recent),
            "change_pct": _r(100 * (ahead - recent) / abs(recent), 1) if recent else None,
            "path": [
                {
                    "date": self.dates[i].isoformat(),
                    "median": _r(self.point[i]),
                    "p10": _r(self.bands["p10"][i]),
                    "p90": _r(self.bands["p90"][i]),
                }
                for i in range(step - 1, h, step)
            ],
            "backtest": self.backtest
            or "not enough history to backtest; treat confidence as unknown",
        }

    def chart(self) -> dict:
        """Payload the frontend draws. Short forecasts get ~4x as much history so
        the fan stays readable; long ones get two years, so yearly cycles (the
        evidence for a seasonal forecast) are on screen."""
        s = self.series
        horizon = len(self.point)
        history_points = 800 if horizon >= LONG_HORIZON else min(365, max(90, 4 * horizon))
        return {
            "series_id": s.id,
            "label": s.label,
            "unit": s.unit,
            "source": s.source_url,
            "history": {
                "dates": [d.isoformat() for d in s.dates[-history_points:]],
                "values": [_r(v) for v in s.values[-history_points:]],
            },
            "forecast": {
                "dates": [d.isoformat() for d in self.dates],
                "median": [_r(v) for v in self.point],
                **{k: [_r(v) for v in vals] for k, vals in self.bands.items()},
            },
            "backtest": self.backtest,
        }


def run_forecast(series: Series, horizon: int, model: Model) -> Forecast:
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be between 1 and {MAX_HORIZON} days")
    if len(series.values) < MIN_HISTORY:
        raise ValueError(
            f"need at least {MIN_HISTORY} points of history, got {len(series.values)}"
        )

    values = np.asarray(series.values, dtype=np.float32)
    positive = bool((values >= 0).all())
    point, deciles = model.predict(values, horizon, positive)

    last = series.dates[-1]
    return Forecast(
        series=series,
        dates=[last + timedelta(days=i + 1) for i in range(horizon)],
        point=[float(v) for v in point],
        bands={k: [float(v) for v in deciles[:, i]] for k, i in _Q.items()},
        backtest=_backtest(values, horizon, positive, model),
    )


def _backtest(values: np.ndarray, horizon: int, positive: bool, model: Model) -> dict | None:
    """Hide the last `horizon` points, forecast them, compare with seasonal-naive."""
    if len(values) < max(MIN_HISTORY, 2 * WEEK) + horizon:
        return None
    train, actual = values[:-horizon], values[-horizon:]
    point, deciles = model.predict(train, horizon, positive)

    naive = np.resize(train[-WEEK:], horizon)  # repeat the last week
    mae_model = float(np.mean(np.abs(point - actual)))
    mae_naive = float(np.mean(np.abs(naive - actual)))
    inside = (actual >= deciles[:, _Q["p10"]]) & (actual <= deciles[:, _Q["p90"]])

    scale = float(np.mean(np.abs(actual))) or 1.0
    return {
        "method": f"held out the last {horizon} days and forecast them blind",
        "model_error_pct_of_typical_value": _r(100 * mae_model / scale, 1),
        "naive_repeat_last_week_error_pct": _r(100 * mae_naive / scale, 1),
        "beats_naive_baseline": mae_model < mae_naive,
        "p10_p90_band_coverage_pct": _r(100 * float(np.mean(inside)), 0),
        "band_coverage_target_pct": 80,
    }


def _r(x: float, digits: int | None = None) -> float:
    """Round for display: fixed digits if given, else 5 significant figures."""
    return round(float(x), digits) if digits is not None else sig(x)
