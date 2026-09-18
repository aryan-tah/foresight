"""The agent loop: Claude reasons and picks tools, TimesFM does the numbers.

`run_turn` is an async generator of plain-dict UI events, so the web layer only
has to forward them. It owns the streaming tool-use loop by hand (rather than
the SDK tool runner) because charts and tool status have to reach the browser
out-of-band, while the turn is still running.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date

import anthropic

from .llm import LLM
from .tools import TOOLS, ToolContext, execute

EFFORT = os.environ.get("FORESIGHT_EFFORT", "medium")
MAX_STEPS = 12

SYSTEM = """\
You are Foresight, an analyst that answers questions about the future with real \
forecasts instead of guesses. You work with a partner: TimesFM 3.0, Google's \
time-series foundation model. You are good at understanding questions, choosing \
evidence and explaining; it is good at numbers. Any number about the future \
comes from the forecast tool, never from you.

How to work:
- Decide which public data series best measures what the user is asking about, \
fetch it, then forecast it. When a question has no direct measure, pick the best \
proxy (for example Wikipedia pageviews for public attention, package downloads \
for developer adoption) and say in one clause that it is a proxy.
- Comparisons ("X vs Y") mean fetching and forecasting each series; do the \
fetches in parallel.
- Choose a horizon that fits the question; 30 days is a sensible default. \
Horizons are in days.
- For anything with a yearly rhythm (holidays, seasons, annual events, \
"when will it peak"), fetch the maximum history (1000 days) so the model sees \
previous cycles, and pick a horizon that reaches past the event. When it \
matters, say what date the forecast peaks on and how that compares with the \
previous years' peaks in the data. Compare honestly: if past peaks sit \
outside the forecast band, say the forecast looks too low or too high rather \
than calling it consistent.
- If a tool fails, read the error and try a corrected name or another source \
before giving up.

How to answer:
- The user already sees a chart of every forecast, so do not repeat tables of \
numbers. Lead with the direct answer in a sentence, then the two or three \
figures that matter (where it is now, where the median goes, how wide the \
p10-p90 band is).
- Be honest about confidence, and ground it in the backtest the forecast tool \
returns: whether the model beat the naive repeat-last-week baseline on the most \
recent period, and whether the 80% band covered about 80% of what actually \
happened. If it lost to the baseline, say the forecast deserves little weight.
- The model only sees the past of this one series. It cannot know about \
upcoming launches, news or shocks; mention that when it matters. Asset prices \
are close to a random walk, so treat price forecasts as a range, not a call, \
and never give financial advice.
- Keep it short: the chart carries the detail. Two brief paragraphs at most, \
around 100 words in total: the answer with its key figures, then how far to \
trust it. Conversational plain text, no headers, lists or markdown.
- Before calling tools, at most one short sentence saying what you will look at.

Today's date is {today}."""


@dataclass
class Session:
    ctx: ToolContext
    messages: list = field(default_factory=list)


async def run_turn(session: Session, user_text: str, llm: LLM) -> AsyncIterator[dict]:
    """Run one user turn to completion, yielding UI events as they happen.

    A failed turn is rolled back, so the history never keeps a half-finished
    exchange (such as a tool call with no result) that would break the next one.
    """
    turn_start = len(session.messages)
    session.messages.append({"role": "user", "content": user_text})

    def fail(message: str) -> dict:
        del session.messages[turn_start:]
        return {"type": "error", "message": message}

    for _ in range(MAX_STEPS):
        try:
            async with llm.client.beta.messages.stream(
                model=llm.model,
                max_tokens=16000,
                system=SYSTEM.format(today=date.today().isoformat()),
                tools=TOOLS,
                messages=session.messages,
                thinking={"type": "adaptive"},
                output_config={"effort": EFFORT},
                cache_control={"type": "ephemeral"},
                **llm.extras,
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield {"type": "text", "text": event.text}
                response = await stream.get_final_message()
        except anthropic.AuthenticationError:
            yield fail(f"{llm.provider} rejected the API key. Check the key in .env.")
            return
        except anthropic.RateLimitError:
            yield fail(f"Rate limited by {llm.provider}. Wait a moment and try again.")
            return
        except anthropic.APIStatusError as e:
            yield fail(f"{llm.provider} API error {e.status_code}: {e.message}")
            return
        except anthropic.APIConnectionError:
            yield fail(f"Could not reach {llm.provider}. Check your connection.")
            return

        session.messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            yield fail("The model declined to answer that one.")
            return
        if response.stop_reason == "max_tokens":
            yield fail("The answer was cut off for length. Try a narrower question.")
            return

        calls = [b for b in response.content if b.type == "tool_use"]
        if not calls:
            yield {"type": "done"}
            return

        for call in calls:
            yield {"type": "tool_start", "id": call.id, "name": call.name, "input": call.input}

        # Independent calls (e.g. two fetches for a comparison) run together,
        # and every result goes back in one user message.
        outcomes = await asyncio.gather(
            *(execute(c.name, c.input, session.ctx) for c in calls)
        )
        results = []
        for call, outcome in zip(calls, outcomes):
            yield {
                "type": "tool_end",
                "id": call.id,
                "ok": not outcome.is_error,
                "note": outcome.note,
            }
            if outcome.chart:
                yield {"type": "chart", "chart": outcome.chart}
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": outcome.content,
                    "is_error": outcome.is_error,
                }
            )
        session.messages.append({"role": "user", "content": results})

    yield fail("Stopped after too many steps without reaching an answer.")
