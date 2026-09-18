from datetime import date

import pytest

from foresight.forecaster import run_forecast
from tests.conftest import make_series


def test_forecast_dates_continue_from_the_last_observation(model):
    f = run_forecast(make_series([100] * 90), 14, model)
    assert f.dates[0] == date(2026, 9, 18)
    assert len(f.dates) == len(f.point) == 14
    assert all(len(b) == 14 for b in f.bands.values())
    assert f.bands["p10"][0] < f.point[0] < f.bands["p90"][0]


def test_backtest_hides_the_last_horizon_points_from_the_model(model):
    run_forecast(make_series(range(100, 190)), 14, model)
    # first call sees everything, the backtest call must see 14 fewer points
    assert [c[0] for c in model.calls] == [90, 76]


def test_backtest_scores_model_against_repeat_last_week():
    class Oracle:
        def __init__(self, truth):
            self.truth = truth

        def predict(self, values, horizon, positive):
            import numpy as np

            point = np.array(self.truth[len(values) : len(values) + horizon], dtype=float)
            if len(point) < horizon:  # the real (non-backtest) call
                point = np.full(horizon, values[-1], dtype=float)
            return point, np.stack([point + d for d in range(-4, 5)], axis=1)

    truth = [100 + 3 * i for i in range(90)]  # steady growth: naive must lose
    bt = run_forecast(make_series(truth), 14, Oracle(truth)).backtest
    assert bt["beats_naive_baseline"] is True
    assert bt["model_error_pct_of_typical_value"] == 0
    assert bt["naive_repeat_last_week_error_pct"] > 0
    assert bt["p10_p90_band_coverage_pct"] == 100


def test_short_history_skips_the_backtest_instead_of_faking_one(model):
    f = run_forecast(make_series([5] * 35), 30, model)
    assert f.backtest is None
    assert "not enough history" in f.summary()["backtest"]


def test_nonnegative_series_are_forecast_as_positive(model):
    run_forecast(make_series([0, 1, 2] * 20), 7, model)
    assert model.calls[0][2] is True
    run_forecast(make_series([-1, 1, 2] * 20), 7, model)
    assert model.calls[-1][2] is False


@pytest.mark.parametrize("horizon", [0, 181])
def test_rejects_out_of_range_horizons(model, horizon):
    with pytest.raises(ValueError, match="horizon"):
        run_forecast(make_series([1] * 90), horizon, model)


def test_rejects_too_little_history(model):
    with pytest.raises(ValueError, match="at least 30"):
        run_forecast(make_series([1] * 10), 7, model)


def test_chart_payload_is_json_ready_and_bounded(model):
    import json

    chart = run_forecast(make_series(range(400)), 30, model).chart()
    json.dumps(chart)
    assert len(chart["history"]["values"]) == 120  # 4x the horizon
    assert len(chart["forecast"]["median"]) == 30
