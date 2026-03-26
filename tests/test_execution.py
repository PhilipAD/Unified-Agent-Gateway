from core.execution import RetryPolicy, RunRecord, RunStore, StepRecord, StepStatus


def test_run_record_last_completed_step_empty():
    run = RunRecord()
    assert run.last_completed_step == -1


def test_run_record_last_completed_step():
    run = RunRecord(
        steps=[
            StepRecord(step_index=0, type="model_call", status=StepStatus.COMPLETED),
            StepRecord(step_index=1, type="tool_execution", status=StepStatus.COMPLETED),
            StepRecord(step_index=2, type="model_call", status=StepStatus.FAILED),
        ]
    )
    assert run.last_completed_step == 1


def test_run_store_save_and_get():
    store = RunStore()
    run = RunRecord(agent_id="a1")
    store.save(run)
    fetched = store.get(run.run_id)
    assert fetched is not None
    assert fetched.agent_id == "a1"


def test_run_store_get_missing():
    store = RunStore()
    assert store.get("nope") is None


def test_run_store_list_runs():
    store = RunStore()
    r1 = RunRecord(agent_id="a")
    r2 = RunRecord(agent_id="b")
    store.save(r1)
    store.save(r2)
    assert len(store.list_runs()) == 2
    assert len(store.list_runs(agent_id="a")) == 1


def test_retry_policy_delay():
    policy = RetryPolicy(base_delay_seconds=1.0, exponential_base=2.0, max_delay_seconds=10.0)
    assert policy.delay_for_attempt(0) == 1.0
    assert policy.delay_for_attempt(1) == 2.0
    assert policy.delay_for_attempt(2) == 4.0
    assert policy.delay_for_attempt(5) == 10.0  # capped at max
