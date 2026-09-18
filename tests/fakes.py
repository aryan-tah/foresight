"""A scripted stand-in for AsyncAnthropic: replays canned assistant turns."""

from types import SimpleNamespace as NS

import anthropic
import httpx


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, input):
    return NS(type="tool_use", id=id, name=name, input=input)


def turn(*blocks, stop_reason=None):
    has_tool = any(b.type == "tool_use" for b in blocks)
    return NS(content=list(blocks), stop_reason=stop_reason or ("tool_use" if has_tool else "end_turn"))


class _Stream:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        if isinstance(self.message, Exception):
            raise self.message
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for block in self.message.content:
            if block.type == "text":
                for word in block.text.split(" "):
                    yield NS(type="text", text=word + " ")

    async def get_final_message(self):
        return self.message


class FakeAnthropic:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kwargs):
        # snapshot: the agent mutates the same list afterwards
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return _Stream(self.script.pop(0))


def auth_error():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)
