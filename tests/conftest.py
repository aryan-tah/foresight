from datetime import date, timedelta

import numpy as np
import pytest

from foresight.series import Series


class FakeModel:
    """Stands in for TimesFM: forecasts a flat line at the last value, +/-10%."""

    def __init__(self):
        self.calls = []

    def predict(self, values, horizon, positive):
        self.calls.append((len(values), horizon, positive))
        point = np.full(horizon, float(values[-1]), dtype=np.float32)
        deciles = np.stack([point * (0.9 + 0.025 * q) for q in range(9)], axis=1)
        return point, deciles


@pytest.fixture
def model():
    return FakeModel()


def make_series(values, series_id="test:series", end=date(2026, 9, 17)):
    n = len(values)
    return Series(
        id=series_id,
        label="Test series",
        unit="things/day",
        dates=[end - timedelta(days=n - 1 - i) for i in range(n)],
        values=[float(v) for v in values],
        source_url="https://example.com",
    )
