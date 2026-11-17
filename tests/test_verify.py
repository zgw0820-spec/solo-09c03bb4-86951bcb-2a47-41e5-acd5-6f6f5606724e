"""Checkpoint verification and quarantine tests.

Covers the restore-audit flow: replay every batch from the saved samples and
training parameters, report the first missing or inconsistent checkpoint,
quarantine the job so workers can neither keep submitting nor take it over,
and never let an interleaved commit turn a verification into a false pass.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import lease as lm
from app import store
from app.config import Settings
from app.db import connect, init_db, transaction
from app.store import JobQuarantined


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "training.db",
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        poll_interval=0.02,
        batch_delay_seconds=0.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def settings(tmp_path):
    s = make_settings(tmp_path)
    s.validate()
    init_db(s.db_path)
    return s


def sample(records: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 3)).tolist()
    labels = [int(i) % 2 for i in range(records)]
    return features, labels


def train_steps(settings, job_id: str, count: int, worker_id: str = "w"):
    lm.register_worker(settings.db_path, worker_id)
    lease = lm.acquire(settings.db_path, job_id, worker_id, settings.lease_seconds)
    for _ in range(count):
        store.advance_job(settings.db_path, lease)
    return lease


def drain(settings, job_id: str, lease) -> None:
    while True:
        if store.advance_job(settings.db_path, lease)["status"] == "completed":
            return


def mutate_checkpoint(settings, job_id: str, step: int, field: str, value) -> None:
    with transaction(settings.db_path) as conn:
        conn.execute(
            f"UPDATE checkpoints SET {field}=? WHERE job_id=? AND step=?",
            (value, job_id, step),
        )


def corrupt_weights_json(raw: str, delta: float = 0.5) -> str:
    values = json.loads(raw)
    values[0] += delta
    return json.dumps(values)


def checkpoint_field(settings, job_id: str, step: int, field: str):
    with connect(settings.db_path) as conn:
        row = conn.execute(
            f"SELECT {field} FROM checkpoints WHERE job_id=? AND step=?",
            (job_id, step),
        ).fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# Healthy jobs
# ---------------------------------------------------------------------------


def test_verify_healthy_job_mid_training(settings):
    features, labels = sample(20, seed=1)
    job = store.create_job(settings.db_path, features, labels, 2)
    train_steps(settings, job["job_id"], 3)

    result = store.verify_job(settings.db_path, job["job_id"])

    assert result["status"] == "ok"
    assert result["resumable"] is True
    assert result["verified_steps"] == 3
    assert result["snapshot_completed_steps"] == 3
    assert result["current_completed_steps"] == 3
    assert result["total_steps"] == 6
    assert result["checkpoints_checked"] == 3
    assert result["first_bad_step"] is None
    assert result["reason"] is None
    assert result["quarantined"] is False
    events = store.list_events(settings.db_path, job["job_id"])
    verified = [e for e in events if e["event"] == "checkpoint_verified"]
    assert verified and verified[-1]["detail"]["verified_steps"] == 3


def test_verify_healthy_completed_job(settings):
    features, labels = sample(16, seed=2)
    job = store.create_job(settings.db_path, features, labels, 2)
    lease = train_steps(settings, job["job_id"], 0)
    drain(settings, job["job_id"], lease)

    result = store.verify_job(settings.db_path, job["job_id"])

    assert result["status"] == "ok"
    assert result["verified_steps"] == result["total_steps"] == 4
    assert store.get_job(settings.db_path, job["job_id"])["quarantined"] is False


def test_verify_queued_job_with_no_checkpoints(settings):
    job = store.create_job(settings.db_path, *sample(8, seed=3), 2)
    result = store.verify_job(settings.db_path, job["job_id"])
    assert result["status"] == "ok"
    assert result["verified_steps"] == 0
    assert result["checkpoints_checked"] == 0


def test_verify_unknown_job_raises_not_found(settings):
    with pytest.raises(store.JobNotFound):
        store.verify_job(settings.db_path, "does-not-exist")


# ---------------------------------------------------------------------------
# Distortion detection
# ---------------------------------------------------------------------------


def test_mid_stream_distortion_detected_even_when_latest_matches(settings):
    """The latest checkpoint and the job row agree with the progress, yet a
    batch in the middle of the stream was distorted - replay must find it."""
    features, labels = sample(20, seed=4)
    job = store.create_job(settings.db_path, features, labels, 3)
    train_steps(settings, job["job_id"], 5)

    tampered = corrupt_weights_json(checkpoint_field(settings, job["job_id"], 3, "weights_json"))
    mutate_checkpoint(settings, job["job_id"], 3, "weights_json", tampered)

    # Sanity: the job row and the latest checkpoint are untouched, so the
    # plain progress query sees nothing wrong.
    current = store.get_job(settings.db_path, job["job_id"])
    assert current["completed_steps"] == 5
    assert current["latest_checkpoint_step"] == 5

    result = store.verify_job(settings.db_path, job["job_id"])

    assert result["status"] == "failed"
    assert result["first_bad_step"] == 3
    assert "weights mismatch at step 3" in result["reason"]
    assert result["verified_steps"] == 2
    assert result["resumable"] is False
    assert result["quarantined"] is True


def test_momentum_distortion_detected(settings):
    job = store.create_job(settings.db_path, *sample(16, seed=5), 2)
    train_steps(settings, job["job_id"], 2)
    tampered = corrupt_weights_json(
        checkpoint_field(settings, job["job_id"], 2, "velocity_json")
    )
    mutate_checkpoint(settings, job["job_id"], 2, "velocity_json", tampered)

    result = store.verify_job(settings.db_path, job["job_id"])

    assert result["status"] == "failed"
    assert result["first_bad_step"] == 2
    assert "momentum mismatch" in result["reason"]


def test_loss_distortion_detected(settings):
    job = store.create_job(settings.db_path, *sample(16, seed=6), 2)
    train_steps(settings, job["job_id"], 2)
    mutate_checkpoint(settings, job["job_id"], 1, "batch_loss", 123.456)

    result = store.verify_job(settings.db_path, job["job_id"])

    assert result["status"] == "failed"
    assert result["first_bad_step"] == 1
    assert "batch_loss mismatch" in result["reason"]

    # A fresh job, this time the dataset loss column.
    job2 = store.create_job(settings.db_path, *sample(16, seed=7), 2)
    train_steps(settings, job2["job_id"], 2)
    mutate_checkpoint(settings, job2["job_id"], 2, "loss", 0.123)

    result2 = store.verify_job(settings.db_path, job2["job_id"])
    assert result2["status"] == "failed"
    assert result2["first_bad_step"] == 2
    assert "loss mismatch" in result2["reason"]


def test_missing_checkpoint_detected(settings):
    job = store.create_job(settings.db_path, *sample(16, seed=8), 2)
    train_steps(settings, job["job_id"], 3)
    with transaction(settings.db_path) as conn:
        conn.execute(
            "DELETE FROM checkpoints WHERE job_id=? AND step=?", (job["job_id"], 2)
        )

    result = store.verify_job(settings.db_path, job["job_id"])

    assert result["status"] == "failed"
    assert result["first_bad_step"] == 2
    assert "missing" in result["reason"]
    assert result["verified_steps"] == 1


def test_unexpected_checkpoint_beyond_progress_detected(settings):
    job = store.create_job(settings.db_path, *sample(16, seed=9), 2)
    train_steps(settings, job["job_id"], 2)
    with transaction(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO checkpoints(job_id, step, weights_json, velocity_json, "
            "loss, batch_loss, committed_at, worker_id, lease_epoch) "
            "VALUES(?, 4, ?, ?, 0.5, 0.5, 0, 'ghost', 1)",
            (job["job_id"], json.dumps([0.0, 0.0, 0.0, 0.0]), json.dumps([0.0, 0.0, 0.0, 0.0])),
        )

    result = store.verify_job(settings.db_path, job["job_id"])

    assert result["status"] == "failed"
    assert result["first_bad_step"] == 4
    assert "unexpected checkpoint" in result["reason"]
    assert result["verified_steps"] == 2


def test_torn_job_row_detected(settings):
    """The job row's weights were rewritten without touching checkpoints."""
    job = store.create_job(settings.db_path, *sample(16, seed=10), 2)
    train_steps(settings, job["job_id"], 2)
    with transaction(settings.db_path) as conn:
        row = conn.execute(
            "SELECT weights_json FROM jobs WHERE job_id=?", (job["job_id"],)
        ).fetchone()
        conn.execute(
            "UPDATE jobs SET weights_json=? WHERE job_id=?",
            (corrupt_weights_json(row["weights_json"]), job["job_id"]),
        )

    result = store.verify_job(settings.db_path, job["job_id"])

    assert result["status"] == "failed"
    assert result["first_bad_step"] == 2
    assert "job weights do not match" in result["reason"]
    # The checkpoint chain itself was fine through step 2.
    assert result["verified_steps"] == 2


def test_corrupted_initial_state_detected(settings):
    job = store.create_job(settings.db_path, *sample(8, seed=11), 2)
    with transaction(settings.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET weights_json=? WHERE job_id=?",
            (json.dumps([1.0, 0.0, 0.0, 0.0]), job["job_id"]),
        )

    result = store.verify_job(settings.db_path, job["job_id"])

    assert result["status"] == "failed"
    assert result["first_bad_step"] == 0
    assert result["verified_steps"] == 0


def test_unparseable_stored_data_is_quarantined_not_raised(settings):
    """A damaged restore (broken JSON) yields a failed verdict, not a 500."""
    job = store.create_job(settings.db_path, *sample(16, seed=24), 2)
    train_steps(settings, job["job_id"], 2)
    mutate_checkpoint(settings, job["job_id"], 2, "weights_json", "{not json")

    result = store.verify_job(settings.db_path, job["job_id"])

    assert result["status"] == "failed"
    assert result["first_bad_step"] == 2
    assert "not parseable" in result["reason"]
    assert result["quarantined"] is True

    # And a job whose own training payload is damaged.
    job2 = store.create_job(settings.db_path, *sample(16, seed=25), 2)
    with transaction(settings.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET features_json=? WHERE job_id=?",
            ("[[1, 2], [3,", job2["job_id"]),
        )
    result2 = store.verify_job(settings.db_path, job2["job_id"])
    assert result2["status"] == "failed"
    assert result2["first_bad_step"] == 0
    assert result2["quarantined"] is True


# ---------------------------------------------------------------------------
# Quarantine semantics
# ---------------------------------------------------------------------------


def quarantine_one_job(settings, seed: int = 12):
    job = store.create_job(settings.db_path, *sample(20, seed=seed), 3)
    lease = train_steps(settings, job["job_id"], 3)
    tampered = corrupt_weights_json(checkpoint_field(settings, job["job_id"], 2, "weights_json"))
    mutate_checkpoint(settings, job["job_id"], 2, "weights_json", tampered)
    result = store.verify_job(settings.db_path, job["job_id"])
    assert result["status"] == "failed"
    return job, lease, result


def test_quarantine_blocks_commits_and_takeover(settings):
    job, lease, result = quarantine_one_job(settings)

    # The worker holding the now-revoked lease cannot commit any more.
    with pytest.raises(JobQuarantined):
        store.advance_job(settings.db_path, lease)

    # The job is no longer claimable, so no worker takes it over.
    assert job["job_id"] not in store.claimable_jobs(settings.db_path)

    # Even a worker that grabs a fresh lease row directly cannot write.
    other = lm.acquire(settings.db_path, job["job_id"], "other", settings.lease_seconds)
    assert other is not None  # the lease row itself is free after quarantine
    with pytest.raises(JobQuarantined):
        store.advance_job(settings.db_path, other)

    state = store.get_job(settings.db_path, job["job_id"])
    assert state["quarantined"] is True
    assert state["completed_steps"] == 3
    assert state["status"] == "running"


def test_late_worker_cannot_rewrite_after_quarantine(settings):
    """A worker suspended before the quarantine and resumed after it must not
    move the job or overwrite the verdict."""
    job = store.create_job(settings.db_path, *sample(16, seed=13), 2)
    lm.register_worker(settings.db_path, "old")
    old = lm.acquire(settings.db_path, job["job_id"], "old", settings.lease_seconds)
    store.advance_job(settings.db_path, old)

    # "old" is frozen here; verification condemns the job meanwhile.
    tampered = corrupt_weights_json(checkpoint_field(settings, job["job_id"], 1, "weights_json"))
    mutate_checkpoint(settings, job["job_id"], 1, "weights_json", tampered)
    verdict = store.verify_job(settings.db_path, job["job_id"])
    assert verdict["status"] == "failed"
    snapshot = store.get_job(settings.db_path, job["job_id"])

    # "old" wakes up and tries to commit; it must be fenced off.
    with pytest.raises(JobQuarantined):
        store.advance_job(settings.db_path, old)
    after = store.get_job(settings.db_path, job["job_id"])
    assert after["completed_steps"] == snapshot["completed_steps"] == 1
    assert after["weights"] == snapshot["weights"]
    assert after["quarantined"] is True

    # Re-verifying reaches the same conclusion and keeps the first quarantine.
    again = store.verify_job(settings.db_path, job["job_id"])
    assert again["status"] == "failed"
    assert again["first_bad_step"] == verdict["first_bad_step"] == 1
    assert again["quarantine_reason"] == verdict["quarantine_reason"]
    quarantined_events = [
        e for e in store.list_events(settings.db_path, job["job_id"])
        if e["event"] == "job_quarantined"
    ]
    assert len(quarantined_events) == 1


def test_quarantined_job_stays_queryable(settings):
    """Even when verification condemned a torn job row / missing head
    checkpoint, status, checkpoints and events remain readable."""
    job = store.create_job(settings.db_path, *sample(16, seed=14), 2)
    train_steps(settings, job["job_id"], 3)
    with transaction(settings.db_path) as conn:
        conn.execute(
            "DELETE FROM checkpoints WHERE job_id=? AND step=3", (job["job_id"],)
        )

    # Before quarantine the hard consistency check trips on this job.
    with pytest.raises(store.StateConflict):
        store.get_job(settings.db_path, job["job_id"])

    result = store.verify_job(settings.db_path, job["job_id"])
    assert result["status"] == "failed"
    assert result["first_bad_step"] == 3

    state = store.get_job(settings.db_path, job["job_id"])
    assert state["quarantined"] is True
    assert state["quarantine_reason"] == result["reason"]
    assert state["quarantined_at"] is not None
    assert state["completed_steps"] == 3
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    assert [c["step"] for c in checkpoints] == [1, 2]
    events = store.list_events(settings.db_path, job["job_id"])
    assert events[-1]["event"] == "job_quarantined"
    assert events[-1]["detail"]["first_bad_step"] == 3


def test_other_jobs_train_normally_while_one_is_quarantined(settings):
    doomed = store.create_job(settings.db_path, *sample(16, seed=15), 2)
    healthy = store.create_job(settings.db_path, *sample(16, seed=16), 2)
    train_steps(settings, doomed["job_id"], 2)
    tampered = corrupt_weights_json(
        checkpoint_field(settings, doomed["job_id"], 1, "weights_json")
    )
    mutate_checkpoint(settings, doomed["job_id"], 1, "weights_json", tampered)
    assert store.verify_job(settings.db_path, doomed["job_id"])["status"] == "failed"

    claimable = store.claimable_jobs(settings.db_path)
    assert claimable == [healthy["job_id"]]
    lease = lm.acquire(settings.db_path, healthy["job_id"], "w", settings.lease_seconds)
    drain(settings, healthy["job_id"], lease)
    final = store.get_job(settings.db_path, healthy["job_id"])
    assert final["status"] == "completed"
    assert final["quarantined"] is False
    assert store.verify_job(settings.db_path, healthy["job_id"])["status"] == "ok"


def test_quarantined_completed_job_hides_final_weights(settings):
    job = store.create_job(settings.db_path, *sample(16, seed=17), 2)
    lease = train_steps(settings, job["job_id"], 0)
    drain(settings, job["job_id"], lease)
    assert store.get_job(settings.db_path, job["job_id"])["final_weights"] is not None

    tampered = corrupt_weights_json(checkpoint_field(settings, job["job_id"], 2, "weights_json"))
    mutate_checkpoint(settings, job["job_id"], 2, "weights_json", tampered)
    result = store.verify_job(settings.db_path, job["job_id"])
    assert result["status"] == "failed"

    state = store.get_job(settings.db_path, job["job_id"])
    assert state["status"] == "completed"  # original status preserved
    assert state["quarantined"] is True
    assert state["final_weights"] is None  # untrustworthy, no longer served


# ---------------------------------------------------------------------------
# Interleaving with batch commits
# ---------------------------------------------------------------------------


def test_stale_when_progress_moves_between_snapshot_and_commit(settings):
    """A commit landing mid-verification must not produce a pass."""
    job = store.create_job(settings.db_path, *sample(16, seed=18), 2)
    lease = train_steps(settings, job["job_id"], 1)

    snapshot = store._verification_snapshot(settings.db_path, job["job_id"])
    store.advance_job(settings.db_path, lease)  # interleaved batch commit
    verdict = store._replay_verdict(snapshot)
    assert verdict["status"] == "ok"  # the snapshot itself was consistent
    result = store._commit_verdict(settings.db_path, snapshot, verdict)

    assert result["status"] == "stale"
    assert result["resumable"] is False
    assert result["snapshot_completed_steps"] == 1
    assert result["current_completed_steps"] == 2

    # Nothing was concluded: no quarantine, no pass recorded.
    state = store.get_job(settings.db_path, job["job_id"])
    assert state["quarantined"] is False
    assert not [
        e for e in store.list_events(settings.db_path, job["job_id"])
        if e["event"] == "checkpoint_verified"
    ]

    # A fresh verification of the quiesced job passes.
    settled = store.verify_job(settings.db_path, job["job_id"])
    assert settled["status"] == "ok"
    assert settled["verified_steps"] == 2


def test_failure_verdict_survives_concurrent_commits(settings):
    """Corruption is in the append-only history: a batch committed between the
    snapshot and the quarantine cannot wash it out."""
    job = store.create_job(settings.db_path, *sample(20, seed=19), 3)
    lease = train_steps(settings, job["job_id"], 2)
    tampered = corrupt_weights_json(checkpoint_field(settings, job["job_id"], 1, "weights_json"))
    mutate_checkpoint(settings, job["job_id"], 1, "weights_json", tampered)

    snapshot = store._verification_snapshot(settings.db_path, job["job_id"])
    store.advance_job(settings.db_path, lease)  # step 3 lands mid-verification
    verdict = store._replay_verdict(snapshot)
    assert verdict["status"] == "failed"
    result = store._commit_verdict(settings.db_path, snapshot, verdict)

    assert result["status"] == "failed"
    assert result["first_bad_step"] == 1
    assert result["snapshot_completed_steps"] == 2
    assert result["current_completed_steps"] == 3
    assert result["quarantined"] is True

    # The lease holder is fenced off at its next commit attempt.
    with pytest.raises(JobQuarantined):
        store.advance_job(settings.db_path, lease)


def test_verification_interleaved_with_a_live_worker_never_false_passes(tmp_path):
    """Stress the snapshot/commit guard: every verdict during live training is
    either a pass for the exact current step or an honest ``stale``."""
    settings = make_settings(tmp_path, batch_delay_seconds=0.02)
    init_db(settings.db_path)
    job = store.create_job(settings.db_path, *sample(24, seed=20), 3)

    from app.worker import Worker

    worker = Worker(settings, "live")
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    saw_ok = False
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            state = store.get_job(settings.db_path, job["job_id"])
            result = store.verify_job(settings.db_path, job["job_id"])
            assert result["status"] in ("ok", "stale")
            assert result["quarantined"] is False
            if result["status"] == "ok":
                saw_ok = True
                # A pass must name the exact progress it proved.
                assert result["verified_steps"] == result["snapshot_completed_steps"]
                assert result["current_completed_steps"] == result["verified_steps"]
            if state["status"] == "completed":
                break
            time.sleep(0.005)
    finally:
        worker.request_stop()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert saw_ok

    final = store.verify_job(settings.db_path, job["job_id"])
    assert final["status"] == "ok"
    assert final["verified_steps"] == final["total_steps"]


def test_worker_stops_submitting_once_quarantined(tmp_path):
    """End to end with a real Worker loop: corrupt, verify, and the running
    worker stops committing while a second job finishes normally."""
    settings = make_settings(tmp_path, batch_delay_seconds=0.05)
    init_db(settings.db_path)
    doomed = store.create_job(settings.db_path, *sample(40, seed=21), 3)
    healthy = store.create_job(settings.db_path, *sample(16, seed=22), 2)

    from app.worker import Worker

    worker = Worker(settings, "w1")
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            if store.get_job(settings.db_path, doomed["job_id"])["completed_steps"] >= 3:
                break
            time.sleep(0.01)
        tampered = corrupt_weights_json(
            checkpoint_field(settings, doomed["job_id"], 2, "weights_json")
        )
        mutate_checkpoint(settings, doomed["job_id"], 2, "weights_json", tampered)
        verdict = store.verify_job(settings.db_path, doomed["job_id"])
        assert verdict["status"] == "failed"
        frozen = store.get_job(settings.db_path, doomed["job_id"])["completed_steps"]

        # Several batch intervals pass; the quarantined job must not move.
        time.sleep(0.4)
        state = store.get_job(settings.db_path, doomed["job_id"])
        assert state["completed_steps"] == frozen
        assert state["quarantined"] is True

        # The healthy job is driven to completion by the same worker.
        deadline = time.time() + 20
        while time.time() < deadline:
            if store.get_job(settings.db_path, healthy["job_id"])["status"] == "completed":
                break
            time.sleep(0.02)
        assert store.get_job(settings.db_path, healthy["job_id"])["status"] == "completed"
    finally:
        worker.request_stop()
        thread.join(timeout=10)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Restored (pre-verification) databases
# ---------------------------------------------------------------------------


def test_restored_database_is_migrated_and_verifiable(tmp_path):
    """A database archived before verification existed gets the new columns
    and can be verified (and quarantined) like any other."""
    db = tmp_path / "restored.db"
    old_schema = """
    CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY, status TEXT NOT NULL, feature_dim INTEGER NOT NULL,
        epochs INTEGER NOT NULL, batch_size INTEGER NOT NULL,
        record_count INTEGER NOT NULL, total_batches INTEGER NOT NULL,
        learning_rate REAL NOT NULL, momentum REAL NOT NULL,
        features_json TEXT NOT NULL, labels_json TEXT NOT NULL,
        weights_json TEXT NOT NULL, velocity_json TEXT NOT NULL,
        loss REAL NOT NULL, completed_steps INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL, updated_at REAL NOT NULL, finished_at REAL
    );
    CREATE TABLE checkpoints (
        job_id TEXT NOT NULL, step INTEGER NOT NULL, weights_json TEXT NOT NULL,
        velocity_json TEXT NOT NULL, loss REAL NOT NULL, batch_loss REAL NOT NULL,
        committed_at REAL NOT NULL, worker_id TEXT NOT NULL,
        lease_epoch INTEGER NOT NULL, PRIMARY KEY (job_id, step)
    );
    CREATE TABLE job_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
        event TEXT NOT NULL, detail TEXT, created_at REAL NOT NULL
    );
    CREATE TABLE workers (
        worker_id TEXT PRIMARY KEY, status TEXT NOT NULL,
        started_at REAL NOT NULL, heartbeat_at REAL NOT NULL
    );
    CREATE TABLE leases (
        job_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL,
        epoch INTEGER NOT NULL, heartbeat_at REAL NOT NULL
    );
    CREATE TABLE lease_epochs (job_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL);
    """
    conn = connect(db)
    try:
        conn.executescript(old_schema)
    finally:
        conn.close()

    init_db(db)  # the migration path every process runs at startup

    settings = make_settings(tmp_path)
    settings = Settings(
        db_path=db,
        lease_seconds=settings.lease_seconds,
        heartbeat_interval=settings.heartbeat_interval,
        poll_interval=settings.poll_interval,
        batch_delay_seconds=settings.batch_delay_seconds,
    )
    job = store.create_job(db, *sample(16, seed=23), 2)
    train_steps(settings, job["job_id"], 2)
    assert store.verify_job(db, job["job_id"])["status"] == "ok"

    tampered = corrupt_weights_json(checkpoint_field(settings, job["job_id"], 1, "weights_json"))
    mutate_checkpoint(settings, job["job_id"], 1, "weights_json", tampered)
    result = store.verify_job(db, job["job_id"])
    assert result["status"] == "failed"
    assert store.get_job(db, job["job_id"])["quarantined"] is True


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = tmp_path / "api.db"
    monkeypatch.setenv("TRAINING_DB_PATH", str(db))
    monkeypatch.setenv("TRAINING_LEASE_SECONDS", "0.6")
    monkeypatch.setenv("TRAINING_HEARTBEAT_INTERVAL", "0.1")

    import app.main as main_module

    monkeypatch.setattr(main_module, "settings", Settings.from_env())
    with TestClient(main_module.app) as test_client:
        yield test_client


def payload(records: int = 16, epochs: int = 2, seed: int = 1):
    rng = np.random.default_rng(seed)
    return {
        "features": rng.normal(size=(records, 2)).tolist(),
        "labels": [i % 2 for i in range(records)],
        "epochs": epochs,
    }


def test_verify_endpoint_ok_and_schema(client):
    job = client.post("/jobs", json=payload()).json()
    settings = Settings.from_env()
    train_steps(settings, job["job_id"], 2)

    response = client.post(f"/jobs/{job['job_id']}/verify")
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["verified_steps"] == 2
    assert result["resumable"] is True

    schema = client.get("/openapi.json").json()
    assert "/jobs/{job_id}/verify" in schema["paths"]
    assert "post" in schema["paths"]["/jobs/{job_id}/verify"]

    assert client.post("/jobs/does-not-exist/verify").status_code == 404


def test_verify_endpoint_quarantines_and_job_stays_queryable(client):
    job = client.post("/jobs", json=payload(seed=5)).json()
    settings = Settings.from_env()
    train_steps(settings, job["job_id"], 2)
    tampered = corrupt_weights_json(checkpoint_field(settings, job["job_id"], 1, "weights_json"))
    mutate_checkpoint(settings, job["job_id"], 1, "weights_json", tampered)

    result = client.post(f"/jobs/{job['job_id']}/verify").json()
    assert result["status"] == "failed"
    assert result["first_bad_step"] == 1
    assert result["quarantined"] is True

    state = client.get(f"/jobs/{job['job_id']}").json()
    assert state["quarantined"] is True
    assert state["quarantine_reason"] == result["reason"]
    assert state["completed_steps"] == 2

    checkpoints = client.get(f"/jobs/{job['job_id']}/checkpoints").json()
    assert [c["step"] for c in checkpoints] == [1, 2]
    events = client.get(f"/jobs/{job['job_id']}/events").json()
    assert events[-1]["event"] == "job_quarantined"

    listed = client.get("/jobs").json()
    assert listed[0]["quarantined"] is True
