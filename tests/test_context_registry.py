import pytest

from context.registry import ContextRegistry, ContextSource, RegisteredContext


@pytest.mark.asyncio
async def test_load_all_basic():
    reg = ContextRegistry()

    async def fetch_static(**kwargs):
        return "static context data"

    reg.register(RegisteredContext(name="static", source=ContextSource.STATIC, fetch=fetch_static))
    result = await reg.load_all()
    assert result == {"static": "static context data"}


@pytest.mark.asyncio
async def test_load_all_truncation():
    reg = ContextRegistry()

    async def fetch_long(**kwargs):
        return "x" * 200

    reg.register(
        RegisteredContext(name="long", source=ContextSource.RAG, fetch=fetch_long, max_chars=50)
    )
    result = await reg.load_all()
    assert len(result["long"]) <= 62  # 50 chars + "\n[truncated]"
    assert result["long"].endswith("[truncated]")


@pytest.mark.asyncio
async def test_load_all_optional_failure_skipped():
    reg = ContextRegistry()

    async def fetch_bad(**kwargs):
        raise RuntimeError("boom")

    reg.register(
        RegisteredContext(name="flaky", source=ContextSource.KV, fetch=fetch_bad, required=False)
    )
    result = await reg.load_all()
    assert "flaky" not in result


@pytest.mark.asyncio
async def test_load_all_required_failure_raises():
    reg = ContextRegistry()

    async def fetch_bad(**kwargs):
        raise RuntimeError("required boom")

    reg.register(
        RegisteredContext(name="critical", source=ContextSource.KV, fetch=fetch_bad, required=True)
    )
    with pytest.raises(RuntimeError, match="required boom"):
        await reg.load_all()
