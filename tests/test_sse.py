import json

from runtime.sse import format_sse


def test_format_sse_basic():
    result = format_sse("chunk", {"type": "chunk", "delta": "hello"})
    lines = result.strip().split("\n")
    assert lines[0] == "event: chunk"
    data = json.loads(lines[1].removeprefix("data: "))
    assert data["delta"] == "hello"


def test_format_sse_done():
    result = format_sse("done", {"type": "done"})
    assert "event: done" in result
    assert result.endswith("\n\n")
