import json

import pytest

from foresight.agent import Session, run_turn
from foresight.tools import ToolContext, execute
from tests.conftest import make_series
from tests.fakes import auth_error, fake_llm, text, tool_use, turn


@pytest.fixture
def session(model):
    s = Session(ToolContext(http=None, model=model))
    s.ctx.store.add(make_series(range(100, 300), "wiki:Thing"))
    return s


async def collect(gen):
    return [e async for e in gen]


async def test_a_turn_streams_text_runs_tools_and_draws_the_chart(session):
    llm = fake_llm([
        turn(text("Checking."), tool_use("t1", "forecast", {"series_id": "wiki:Thing", "horizon_days": 14})),
        turn(text("It keeps rising.")),
    ])
    events = await collect(run_turn(session, "Will Thing grow?", llm))

    kinds = [e["type"] for e in events]
    assert kinds.index("tool_start") < kinds.index("tool_end") < kinds.index("chart")
    assert kinds[-1] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "text").strip() == "Checking. It keeps rising."
    chart = next(e["chart"] for e in events if e["type"] == "chart")
    assert chart["series_id"] == "wiki:Thing" and len(chart["forecast"]["median"]) == 14

    # the second request carries the tool result back, as one user message
    result_msg = llm.client.requests[1]["messages"][-1]
    assert result_msg["role"] == "user"
    [result] = result_msg["content"]
    assert result["tool_use_id"] == "t1" and result["is_error"] is False
    assert json.loads(result["content"])["horizon_days"] == 14


async def test_parallel_tool_calls_return_all_results_in_a_single_message(session):
    session.ctx.store.add(make_series(range(50, 250), "wiki:Other"))
    llm = fake_llm([
        turn(
            tool_use("a", "forecast", {"series_id": "wiki:Thing", "horizon_days": 7}),
            tool_use("b", "forecast", {"series_id": "wiki:Other", "horizon_days": 7}),
        ),
        turn(text("Thing wins.")),
    ])
    events = await collect(run_turn(session, "Thing vs Other?", llm))
    assert sum(e["type"] == "chart" for e in events) == 2
    results = llm.client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


async def test_tool_failures_go_back_to_the_model_as_errors_not_crashes(session):
    llm = fake_llm([
        turn(tool_use("t1", "forecast", {"series_id": "wiki:Missing", "horizon_days": 7})),
        turn(text("I could not find that series.")),
    ])
    events = await collect(run_turn(session, "?", llm))
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["ok"] is False
    [result] = llm.client.requests[1]["messages"][-1]["content"]
    assert result["is_error"] is True
    assert "wiki:Thing" in result["content"]  # tells the model what it can use


async def test_a_failed_turn_is_rolled_back_so_the_next_one_still_works(session):
    llm = fake_llm([auth_error(), turn(text("Hello."))])
    events = await collect(run_turn(session, "first", llm))
    assert events == [{"type": "error", "message": events[0]["message"]}]
    assert "API key" in events[0]["message"]
    assert session.messages == []

    await collect(run_turn(session, "second", llm))
    assert [m["role"] for m in session.messages] == ["user", "assistant"]


async def test_a_truncated_tool_call_is_never_executed(session, model):
    llm = fake_llm([
        turn(tool_use("t1", "forecast", {"series_id": "wiki:Thing"}), stop_reason="max_tokens"),
    ])
    events = await collect(run_turn(session, "?", llm))
    assert events[-1]["type"] == "error"
    assert model.calls == [] and session.messages == []


async def test_runaway_tool_loops_are_cut_off(session):
    looping = [turn(tool_use(f"t{i}", "forecast", {"series_id": "wiki:Thing", "horizon_days": 7})) for i in range(20)]
    events = await collect(run_turn(session, "?", fake_llm(looping)))
    assert events[-1]["type"] == "error" and "too many steps" in events[-1]["message"]


async def test_bad_tool_inputs_are_reported_not_raised(session):
    assert (await execute("forecast", {"series_id": "wiki:Thing"}, session.ctx)).is_error
    assert (await execute("forecast", {"series_id": "wiki:Thing", "horizon_days": 999}, session.ctx)).is_error
    assert (await execute("no_such_tool", {}, session.ctx)).is_error
    assert (await execute("forecast", "not a dict", session.ctx)).is_error


async def test_provider_specific_request_options_reach_the_api(session):
    llm = fake_llm([turn(text("Hi."))], fallbacks="default")
    await collect(run_turn(session, "hi", llm))
    assert llm.client.requests[0]["fallbacks"] == "default"
    assert llm.client.requests[0]["model"] == "test-model"
