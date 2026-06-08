from __future__ import annotations

import logging
import math
import threading
from types import SimpleNamespace

import torch

from lerobot.rl.rlt_buffer import RLTReplayBuffer, RLTReplaySample
from tests.utils import require_package


def _sample(offset: float = 0.0) -> RLTReplaySample:
    return RLTReplaySample(
        rl_token=torch.ones(4) + offset,
        proprio=torch.ones(2) + offset,
        reference_chunk=torch.zeros(3, 2) + offset,
        executed_chunk=torch.ones(3, 2) + offset,
        next_rl_token=torch.ones(4) * 2 + offset,
        next_proprio=torch.ones(2) * 2 + offset,
        next_reference_chunk=torch.zeros(3, 2) + offset,
        reward=1.0,
        done=False,
        is_intervention=True,
    )


@require_package("grpcio", "grpc")
def test_load_rlt_replay_file_emits_status_with_replay_size(tmp_path, monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    replay_path = tmp_path / "rlt_online_replay.pt"
    replay = RLTReplayBuffer(capacity=8)
    replay.add(_sample(0.0))
    replay.add(_sample(1.0))
    replay.save(replay_path)

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        policy_server_drtc,
        "emit_status",
        lambda source, event, **fields: emitted.append((source, event, fields)),
    )

    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server._rlt_replay_lock = threading.Lock()
    server._rlt_replay_capacity = 8
    server._rlt_replay = RLTReplayBuffer(capacity=8)
    server._rlt_completed_episodes = set()
    server._rlt_train_step = 0
    server._rlt_online_collection_enabled = True
    server._rlt_online_training_enabled = True
    server._rlt_training_head = "idle"
    server._rlt_actor_disabled_by_safety = False
    server._rlt_demo_replay_size = 0
    server._rlt_online_replay_size = 0
    server._rlt_accepted_transitions = 0
    server._rlt_accepted_frames = 0
    server._rlt_episode_id_offset = 0
    server.logger = logging.getLogger("test_policy_server_drtc")

    loaded_size = server._load_rlt_replay_file(str(replay_path), source="online")

    assert loaded_size == 2
    assert len(server._rlt_replay) == 2
    assert server._rlt_episode_id_offset == 0
    assert emitted[0][0] == "policy_server"
    assert emitted[0][1] == "rlt_replay_loaded"
    assert emitted[0][2]["rlt_replay_size"] == 2
    assert emitted[0][2]["rlt_episode_id_offset"] == 0
    assert emitted[0][2]["replay_loaded_size"] == 2
    assert emitted[0][2]["replay_source"] == "online"
    assert emitted[0][2]["replay_path"] == str(replay_path)


@require_package("grpcio", "grpc")
def test_online_replay_load_sets_episode_id_offset(tmp_path, monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    replay_path = tmp_path / "rlt_online_replay.pt"
    replay = RLTReplayBuffer(capacity=8)
    old_sample = _sample(0.0)
    old_sample.episode_id = 41
    replay.add(old_sample)
    replay.save(replay_path)

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)

    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server._rlt_replay_lock = threading.Lock()
    server._rlt_replay_capacity = 8
    server._rlt_replay = RLTReplayBuffer(capacity=8)
    server._rlt_completed_episodes = set()
    server._rlt_train_step = 0
    server._rlt_online_collection_enabled = True
    server._rlt_online_training_enabled = True
    server._rlt_training_head = "idle"
    server._rlt_actor_disabled_by_safety = False
    server._rlt_demo_replay_size = 0
    server._rlt_online_replay_size = 0
    server._rlt_accepted_transitions = 0
    server._rlt_accepted_frames = 0
    server._rlt_episode_id_offset = 0
    server.logger = logging.getLogger("test_policy_server_drtc")

    assert server._load_rlt_replay_file(str(replay_path), source="online") == 1
    assert server._rlt_episode_id_offset == 41


@require_package("grpcio", "grpc")
def test_demo_replay_fraction_reserves_demo_slots_and_rolls_online_fifo(tmp_path, monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    demo_path = tmp_path / "rlt_demo_replay.pt"
    online_path = tmp_path / "rlt_online_replay.pt"
    demo_replay = RLTReplayBuffer(capacity=16)
    online_replay = RLTReplayBuffer(capacity=16)
    for offset in range(8):
        demo_replay.add(_sample(float(offset)))
    for offset in range(8):
        online_replay.add(_sample(float(100 + offset)))
    demo_replay.save(demo_path)
    online_replay.save(online_path)

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)

    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server._rlt_replay_lock = threading.Lock()
    server._rlt_replay_capacity = 8
    server._rlt_demo_replay_fraction = 0.25
    server._rlt_replay = RLTReplayBuffer(capacity=8)
    server._rlt_demo_replay_samples = []
    server._rlt_online_replay = RLTReplayBuffer(capacity=8)
    server._rlt_completed_episodes = set()
    server._rlt_train_step = 0
    server._rlt_online_collection_enabled = True
    server._rlt_online_training_enabled = True
    server._rlt_training_head = "idle"
    server._rlt_actor_disabled_by_safety = False
    server._rlt_demo_replay_size = 0
    server._rlt_online_replay_size = 0
    server._rlt_accepted_transitions = 0
    server._rlt_accepted_frames = 0
    server._rlt_episode_id_offset = 0
    server.logger = logging.getLogger("test_policy_server_drtc")

    assert server._load_rlt_replay_file(str(demo_path), source="demo") == 8
    assert server._load_rlt_replay_file(str(online_path), source="online") == 6

    training_offsets = [float(sample.rl_token[0].item() - 1.0) for sample in server._rlt_replay.samples()]
    demo_offsets = [offset for offset in training_offsets if offset < 100]
    online_offsets = [offset for offset in training_offsets if offset >= 100]

    assert len(server._rlt_replay) == 8
    assert len(server._rlt_demo_replay_samples) == 8
    assert len(server._rlt_online_replay) == 6
    assert len(demo_offsets) == 2
    assert online_offsets == [102.0, 103.0, 104.0, 105.0, 106.0, 107.0]


@require_package("grpcio", "grpc")
def test_load_rlt_review_archive_preserves_existing_samples(tmp_path, monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    archive_path = tmp_path / "rlt_review_archive.pt"
    replay = RLTReplayBuffer(capacity=8)
    replay.add(_sample(0.0))
    archived_sample = _sample(1.0)
    archived_sample.episode_id = 12
    replay.add(archived_sample)
    replay.save(archive_path)

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)

    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server._rlt_review_archive_path = str(archive_path)
    server._rlt_review_archive = []
    server._rlt_replay_lock = threading.Lock()
    server._rlt_replay = RLTReplayBuffer(capacity=8)
    server._rlt_replay_capacity = 8
    server._rlt_completed_episodes = set()
    server._rlt_train_step = 0
    server._rlt_online_collection_enabled = True
    server._rlt_online_training_enabled = True
    server._rlt_training_head = "idle"
    server._rlt_actor_disabled_by_safety = False
    server._rlt_demo_replay_size = 0
    server._rlt_online_replay_size = 0
    server._rlt_accepted_transitions = 0
    server._rlt_accepted_frames = 0
    server._rlt_episode_id_offset = 0
    server.logger = logging.getLogger("test_policy_server_drtc")

    server._load_rlt_review_archive()

    assert len(server._rlt_review_archive) == 2
    assert server._rlt_episode_id_offset == 12


class _DiagnosticStub:
    def counter(self, *_args, **_kwargs):
        pass

    def set_context(self, **_kwargs):
        pass


def _server_for_rlt_status(policy_server_drtc):
    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server.policy_type = "molmoact2_rlt"
    server._rlt_replay_lock = threading.Lock()
    server._rlt_replay = RLTReplayBuffer(capacity=4)
    server._rlt_replay_capacity = 4
    server._rlt_completed_episodes = set()
    server._rlt_train_step = 0
    server._rlt_loaded_head_step = 0
    server._rlt_execute_after_train_steps = 1000
    server._rlt_online_collection_enabled = True
    server._rlt_online_training_enabled = True
    server._rlt_training_operator_enabled = True
    server._rlt_actor_operator_enabled = True
    server._rlt_actor_critical_phase_active = False
    server._rlt_eval_actor_blend = 1.0
    server._rlt_training_head = "idle"
    server._rlt_actor_disabled_by_safety = False
    server._rlt_demo_replay_size = 0
    server._rlt_online_replay_size = 0
    server._rlt_accepted_transitions = 0
    server._rlt_accepted_frames = 0
    server._rlt_critic_updates_per_actor = 1
    server._rlt_success_sample_fraction = 0.0
    server._rlt_intervention_sample_fraction = 0.0
    server._rlt_intervention_reference_mode = "executed"
    server._rlt_wandb_enabled = False
    server._rlt_wandb_run = None
    server._rlt_wandb_project = "lerobot-rlt"
    server._rlt_wandb_run_name = None
    server._rlt_wandb_mode = None
    server._rlt_last_policy_mode = "not_configured"
    server._rlt_last_actor_executing = False
    server._rlt_last_actor_gate_reason = "not_configured"
    server._rlt_last_actor_prediction_available = False
    server._rlt_last_action_deviation_rms = None
    server._rlt_last_action_deviation_abs_max = None
    server._rlt_last_inference_event_ts = None
    server._rlt_last_inference_status_key = None
    server._rlt_last_inference_status_ts = 0.0
    return server


@require_package("grpcio", "grpc")
def test_rlt_status_reports_persisted_head_checkpoint(monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        policy_server_drtc,
        "emit_status",
        lambda source, event, **fields: emitted.append((source, event, fields)),
    )

    server = _server_for_rlt_status(policy_server_drtc)
    server.policy = SimpleNamespace(
        config=SimpleNamespace(
            rlt_enabled=True,
            rlt_embedding_checkpoint="outputs/embed.pt",
            rlt_head_checkpoint="outputs/rlt_head_latest.pt",
        ),
        _rlt_actor_loaded=True,
        _rlt_loaded_head_step=250,
    )
    server._rlt_loaded_head_step = 250
    server._rlt_train_step = 250

    server._emit_rlt_status("rlt_configured")

    fields = emitted[0][2]
    assert fields["rlt_enabled"] is True
    assert fields["rlt_head_checkpoint"] == "outputs/rlt_head_latest.pt"
    assert fields["rlt_head_checkpoint_loaded"] is True
    assert fields["rlt_head_status"] == "loaded_from_disk"
    assert fields["rlt_actor_available"] is True
    assert fields["rlt_loaded_head_step"] == 250


@require_package("grpcio", "grpc")
def test_rlt_inference_status_persists_actor_usage_and_vla_delta(monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        policy_server_drtc,
        "emit_status",
        lambda source, event, **fields: emitted.append((source, event, fields)),
    )

    server = _server_for_rlt_status(policy_server_drtc)
    server.policy = SimpleNamespace(
        config=SimpleNamespace(rlt_enabled=True, rlt_head_checkpoint=None),
        _rlt_actor_loaded=True,
        _rlt_loaded_head_step=0,
    )

    server._emit_rlt_inference_status(
        policy_mode="rlt_actor",
        critical_phase_active=True,
        actor_executing=True,
        action_deviation_rms=0.125,
        action_deviation_abs_max=0.5,
        window_start_index=2,
        window_len=10,
        rollout_id=7,
        critical_phase_id=3,
        rollout_open=True,
    )
    server._emit_rlt_inference_status(
        policy_mode="rlt_actor",
        critical_phase_active=True,
        actor_executing=True,
        action_deviation_rms=0.125,
        action_deviation_abs_max=0.5,
        window_start_index=2,
        window_len=10,
        rollout_id=7,
        critical_phase_id=4,
        rollout_open=True,
    )
    server._emit_rlt_status("rlt_training_state")

    inference_fields = emitted[0][2]
    assert inference_fields["rlt_policy_mode"] == "rlt_actor"
    assert inference_fields["rlt_actor_executing"] is True
    assert inference_fields["rlt_actor_gate_reason"] == "executing"
    assert inference_fields["rlt_action_deviation_rms"] == 0.125
    assert inference_fields["rlt_action_deviation_abs_max"] == 0.5
    assert inference_fields["rollout_id"] == 7
    assert inference_fields["critical_phase_id"] == 3
    assert inference_fields["rollout_open"] is True
    assert inference_fields["critical_phase_open"] is True

    next_inference_fields = emitted[1][2]
    assert next_inference_fields["critical_phase_id"] == 4

    later_fields = emitted[2][2]
    assert later_fields["rlt_policy_mode"] == "rlt_actor"
    assert later_fields["rlt_actor_executing"] is True
    assert later_fields["rlt_action_deviation_rms"] == 0.125
    assert later_fields["rlt_action_deviation_abs_max"] == 0.5


@require_package("grpcio", "grpc")
def test_rlt_status_reports_replay_outcome_mix_and_sampling_weights(monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        policy_server_drtc,
        "emit_status",
        lambda source, event, **fields: emitted.append((source, event, fields)),
    )

    server = _server_for_rlt_status(policy_server_drtc)
    success = _sample(0.0)
    success.success = True
    success.failure = False
    success.reward = 1.0
    failure = _sample(1.0)
    failure.success = False
    failure.failure = True
    failure.reward = 0.0
    server._rlt_replay.extend([success, failure])
    server._rlt_success_sample_fraction = 0.6
    server._rlt_intervention_sample_fraction = 0.1

    server._emit_rlt_status("rlt_training_state")

    fields = emitted[0][2]
    assert fields["rlt_replay_sample_success_fraction"] == 0.5
    assert fields["rlt_replay_sample_failure_fraction"] == 0.5
    assert fields["rlt_replay_sample_success_count"] == 1.0
    assert fields["rlt_replay_sample_failure_count"] == 1.0
    assert fields["rlt_success_sample_fraction"] == 0.6
    assert fields["rlt_failure_sample_fraction"] == 0.3
    assert fields["rlt_intervention_sample_fraction"] == 0.1


def _server_for_accept_transition(policy_server_drtc):
    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server._rlt_context_cache = policy_server_drtc.RLTSourceContextCache(max_size=4)
    server._rlt_next_context_id = 1
    server._rlt_replay_lock = threading.Lock()
    server._rlt_replay = RLTReplayBuffer(capacity=4)
    server._rlt_replay_capacity = 4
    server._rlt_review_archive_path = None
    server._rlt_review_archive = []
    server._rlt_train_step = 0
    server._rlt_online_collection_enabled = True
    server._rlt_online_training_enabled = True
    server._rlt_training_head = "idle"
    server._rlt_actor_disabled_by_safety = False
    server._rlt_demo_replay_size = 0
    server._rlt_accepted_transitions = 0
    server._rlt_accepted_frames = 0
    server._rlt_recent_episode_successes = policy_server_drtc.deque(maxlen=20)
    server._rlt_episode_success_count = 0
    server._rlt_episode_failure_count = 0
    server._rlt_episode_open_count = 0
    server._rlt_last_raw_outcome = "open"
    server._rlt_last_raw_outcome_success = 0.0
    server._rlt_last_raw_outcome_failure = 0.0
    server._rlt_last_raw_outcome_open = 0.0
    server._rlt_online_replay_size = 0
    server._rlt_online_buffer_save_freq_transitions = 0
    server._rlt_buffer_dirty = False
    server._rlt_completed_episodes = set()
    server._rlt_episode_id_offset = 0
    server._rlt_intervention_reference_mode = "executed"
    server._action_encoding = "raw"
    server._action_normalizer = None
    server._metrics = SimpleNamespace(diagnostic=_DiagnosticStub())
    return server


@require_package("grpcio", "grpc")
def test_rlt_effective_bc_beta_uses_exponential_decay_with_floor():
    from lerobot.async_inference import policy_server_drtc

    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server.policy = SimpleNamespace(config=SimpleNamespace(rlt_bc_beta=0.95))
    server._rlt_train_step = 0
    server._rlt_bc_beta_initial = 0.95
    server._rlt_bc_beta_decay_steps = 600.0
    server._rlt_bc_beta_min = 0.01

    assert math.isclose(server._rlt_effective_bc_beta(0), 0.95)
    assert math.isclose(
        server._rlt_effective_bc_beta(600),
        0.01 + (0.95 - 0.01) * math.exp(-1.0),
    )
    assert server._rlt_effective_bc_beta(100_000) >= 0.01
    assert math.isclose(server._rlt_effective_bc_beta(100_000), 0.01, rel_tol=0.0, abs_tol=1e-6)


@require_package("grpcio", "grpc")
def test_rlt_replay_outcome_stats_count_success_failure_and_open():
    from lerobot.async_inference import policy_server_drtc

    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server._rlt_replay_capacity = 8
    server._rlt_replay = RLTReplayBuffer(capacity=8)

    success = _sample(0.0)
    success.episode_id = 1
    success.success = True
    success.reward = 1.0
    failure = _sample(1.0)
    failure.episode_id = 2
    failure.success = False
    failure.failure = True
    failure.reward = 0.0
    open_sample = _sample(2.0)
    open_sample.episode_id = 3
    open_sample.success = False
    open_sample.failure = False
    open_sample.reward = 0.0
    server._rlt_replay.extend([success, failure, open_sample])

    stats = server._rlt_replay_outcome_stats_locked()

    assert stats["rlt_replay_sample_success_count"] == 1.0
    assert stats["rlt_replay_sample_failure_count"] == 1.0
    assert stats["rlt_replay_sample_open_count"] == 1.0
    assert math.isclose(stats["rlt_replay_sample_success_fraction"], 1 / 3)
    assert stats["rlt_replay_episode_success_count"] == 1.0
    assert stats["rlt_replay_episode_failure_count"] == 1.0
    assert stats["rlt_replay_episode_open_count"] == 1.0


@require_package("grpcio", "grpc")
def test_rlt_outcome_wandb_stats_track_raw_and_moving_success():
    from lerobot.async_inference import policy_server_drtc

    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server._rlt_recent_episode_successes = policy_server_drtc.deque(maxlen=3)
    server._rlt_episode_success_count = 0
    server._rlt_episode_failure_count = 0
    server._rlt_episode_open_count = 0
    server._rlt_last_raw_outcome = "open"
    server._rlt_last_raw_outcome_success = 0.0
    server._rlt_last_raw_outcome_failure = 0.0
    server._rlt_last_raw_outcome_open = 0.0

    server._record_rlt_transition_outcome(
        SimpleNamespace(success=True, failure=False, reward=1.0, done=True)
    )
    server._record_rlt_transition_outcome(
        SimpleNamespace(success=False, failure=True, reward=0.0, done=True)
    )
    server._record_rlt_transition_outcome(
        SimpleNamespace(success=False, failure=False, reward=0.0, done=True)
    )

    stats = server._rlt_outcome_wandb_stats()

    assert stats["rlt_raw_outcome"] == "open"
    assert stats["rlt_raw_outcome_success"] == 0.0
    assert stats["rlt_raw_outcome_failure"] == 0.0
    assert stats["rlt_raw_outcome_open"] == 1.0
    assert stats["rlt_episode_success_count"] == 1.0
    assert stats["rlt_episode_failure_count"] == 1.0
    assert stats["rlt_episode_open_count"] == 1.0
    assert stats["rlt_episode_count"] == 3.0
    assert math.isclose(stats["rlt_success_moving_avg"], 1 / 3)
    assert stats["rlt_success_moving_avg_count"] == 3.0


@require_package("grpcio", "grpc")
def test_cache_rlt_source_context_uses_shifted_window(monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)
    server = _server_for_accept_transition(policy_server_drtc)
    server.policy = SimpleNamespace(config=SimpleNamespace(rlt_chunk_size=3))
    server._rlt_review_capture_enabled = False
    server._rlt_image_encoder = None

    reference = torch.arange(10, dtype=torch.float32).view(1, 5, 2)
    context_id = server._cache_rlt_source_context(
        source_control_step=50,
        chunk_start_step=100,
        reference=reference,
        rl_token=torch.ones(1, 4),
        proprio=torch.ones(1, 2),
        anchor_state=None,
        window_start_index=2,
        rlt_checkpoint_step=123,
        rlt_policy_mode="vla_passthrough",
        rlt_actor_executing=False,
        rollout_id=2,
        critical_phase_id=7,
    )

    cached = server._rlt_context_cache.get(context_id)
    assert cached is not None
    assert cached.chunk_start_step == 102
    assert cached.rlt_checkpoint_step == 123
    assert cached.policy_origin == "base_vla"
    assert cached.rlt_policy_mode == "vla_passthrough"
    assert cached.rlt_actor_executing is False
    assert cached.rollout_id == 2
    assert cached.critical_phase_id == 7
    assert torch.equal(cached.reference_chunk, reference[:, 2:5].squeeze(0))


@require_package("grpcio", "grpc")
def test_accept_rlt_transition_uses_executed_chunk_as_intervention_reference(monkeypatch):
    from lerobot.async_inference import policy_server_drtc
    from lerobot.transport import services_pb2

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)

    server = _server_for_accept_transition(policy_server_drtc)

    source_reference = torch.zeros(3, 2)
    source = policy_server_drtc.RLTSourceContext(
        context_id=1,
        source_control_step=10,
        chunk_start_step=10,
        rl_token=torch.ones(4),
        proprio=torch.ones(2),
        reference_chunk=source_reference,
        anchor_state=None,
        rlt_checkpoint_step=250,
    )
    next_context = policy_server_drtc.RLTSourceContext(
        context_id=2,
        source_control_step=13,
        chunk_start_step=13,
        rl_token=torch.ones(4) * 2,
        proprio=torch.ones(2) * 2,
        reference_chunk=torch.ones(3, 2) * 3,
        anchor_state=None,
    )
    server._rlt_context_cache.put(source)
    server._rlt_context_cache.put(next_context)

    executed = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
    transition = services_pb2.RLTTransitionChunk(
        episode_id=7,
        source_rlt_context_id=1,
        next_rlt_context_id=2,
        chunk_start_step=10,
        num_actions=3,
        action_dim=2,
        executed_actions_f32=executed.numpy().tobytes(),
        reward=0.0,
        done=False,
        is_intervention=True,
    )

    server._accept_rlt_transition(transition)

    sample = server._rlt_replay.samples()[0]
    assert torch.equal(sample.reference_chunk, executed)
    assert torch.equal(sample.executed_chunk, executed)
    assert torch.equal(sample.next_reference_chunk, next_context.reference_chunk)
    assert sample.rlt_checkpoint_step == 250
    assert server._rlt_accepted_frames == 3


@require_package("grpcio", "grpc")
def test_accept_rlt_transition_persists_policy_origin(monkeypatch):
    from lerobot.async_inference import policy_server_drtc
    from lerobot.transport import services_pb2

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        policy_server_drtc,
        "emit_status",
        lambda source, event, **fields: emitted.append((source, event, fields)),
    )

    server = _server_for_accept_transition(policy_server_drtc)

    source = policy_server_drtc.RLTSourceContext(
        context_id=1,
        source_control_step=10,
        chunk_start_step=10,
        rl_token=torch.ones(4),
        proprio=torch.ones(2),
        reference_chunk=torch.ones(3, 2),
        anchor_state=None,
        policy_origin="rlt_head",
        rlt_policy_mode="rlt_actor",
        rlt_actor_executing=True,
        rollout_id=5,
        critical_phase_id=7,
    )
    server._rlt_context_cache.put(source)

    executed = torch.zeros(3, 2, dtype=torch.float32)
    transition = services_pb2.RLTTransitionChunk(
        episode_id=7,
        source_rlt_context_id=1,
        next_rlt_context_id=0,
        chunk_start_step=10,
        num_actions=3,
        action_dim=2,
        executed_actions_f32=executed.numpy().tobytes(),
        reward=1.0,
        done=True,
        is_intervention=False,
        success=True,
        failure=False,
    )

    server._accept_rlt_transition(transition)

    sample = server._rlt_replay.samples()[0]
    assert sample.policy_origin == "rlt_head"
    assert sample.rlt_policy_mode == "rlt_actor"
    assert sample.rlt_actor_executing is True
    assert sample.rollout_id == 5
    assert sample.critical_phase_id == 7

    accepted = next(fields for _source, event, fields in emitted if event == "rlt_transition_accepted")
    assert accepted["policy_origin"] == "rlt_head"
    assert accepted["rlt_policy_mode"] == "rlt_actor"
    assert accepted["rlt_actor_executing"] is True
    assert accepted["rollout_id"] == 5
    assert accepted["critical_phase_id"] == 7


@require_package("grpcio", "grpc")
def test_rlt_training_control_toggle_enables_operator(monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)

    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server.policy_type = "molmoact2_rlt"
    server._rlt_replay_lock = threading.Lock()
    server._rlt_replay = RLTReplayBuffer(capacity=4)
    server._rlt_replay_capacity = 4
    server._rlt_completed_episodes = set()
    server._rlt_train_step = 0
    server._rlt_online_collection_enabled = True
    server._rlt_online_training_enabled = True
    server._rlt_training_head = "paused"
    server._rlt_training_operator_enabled = False
    server._rlt_actor_disabled_by_safety = False
    server._rlt_demo_replay_size = 0
    server._rlt_online_replay_size = 0
    server._rlt_accepted_transitions = 0
    server._rlt_accepted_frames = 0
    server._rlt_warmup_episodes = 0
    server._rlt_warmup_transitions = 0
    server._rlt_batch_size = 1
    server._rlt_actor_optimizer = object()
    server._rlt_critic_optimizer = object()
    server._rlt_head_status_fields = lambda: {}
    server._poll_rlt_override_file = lambda: None
    server._tui_control_reader = SimpleNamespace(read_events=lambda: [{"command": "toggle_rlt_training"}])

    server._poll_rlt_training_controls()

    assert server._rlt_training_operator_enabled is True
    assert server._rlt_training_head == "idle"


@require_package("grpcio", "grpc")
def test_rlt_success_reward_boost_updates_live_replay(monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        policy_server_drtc,
        "emit_status",
        lambda source, event, **fields: emitted.append((source, event, fields)),
    )

    server = _server_for_rlt_status(policy_server_drtc)
    server.policy = SimpleNamespace(config=SimpleNamespace(rlt_enabled=True, rlt_head_checkpoint=None))
    server._rlt_demo_replay_samples = []
    server._rlt_demo_replay_fraction = 0.0
    server._rlt_online_replay = RLTReplayBuffer(capacity=4)
    server._rlt_online_buffer_path = None
    server._rlt_review_archive_path = None
    server._rlt_review_archive = []
    server._rlt_review_archive_dirty = False
    server._rlt_buffer_dirty = False

    first = _sample(0.0)
    first.episode_id = 12
    first.reward = 0.0
    first.done = False
    first.success = True
    second = _sample(1.0)
    second.episode_id = 12
    second.reward = 1.0
    second.done = True
    second.success = True
    server._rlt_online_replay.extend([first, second])
    server._rebuild_rlt_training_replay_locked()

    server._apply_rlt_success_reward_boost(
        {
            "command": "boost_success_reward",
            "server_episode_id": 12,
            "critical_phase_id": 3,
            "rollout_id": 2,
        }
    )

    online_samples = server._rlt_online_replay.samples()
    training_samples = server._rlt_replay.samples()
    assert [sample.reward for sample in online_samples] == [0.0, 2.0]
    assert [sample.done for sample in online_samples] == [False, True]
    assert [sample.success for sample in online_samples] == [True, True]
    assert [sample.reward for sample in training_samples] == [0.0, 2.0]
    assert server._rlt_buffer_dirty is True
    assert emitted[-1][1] == "rlt_critical_success_reward_boosted"
    assert emitted[-1][2]["episode_id"] == 12
    assert emitted[-1][2]["critical_phase_id"] == 3
    assert emitted[-1][2]["reward"] == 2.0
    assert emitted[-1][2]["reward_boosted"] is True


@require_package("grpcio", "grpc")
def test_rlt_actor_execution_is_gated_to_operator_enabled_critical_phase():
    from lerobot.async_inference import policy_server_drtc

    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server.policy_type = "molmoact2_rlt"
    server.policy = SimpleNamespace(
        config=SimpleNamespace(rlt_enabled=True),
        _rlt_actor_loaded=True,
    )
    server._rlt_actor_disabled_by_safety = False
    server._rlt_actor_operator_enabled = True
    server._rlt_train_step = 0
    server._rlt_execute_after_train_steps = 1000

    assert server._rlt_should_execute_actor(critical_phase_active=True)
    assert not server._rlt_should_execute_actor(critical_phase_active=False)

    server._rlt_actor_operator_enabled = False
    assert not server._rlt_should_execute_actor(critical_phase_active=True)


@require_package("grpcio", "grpc")
def test_accept_rlt_transition_can_keep_original_intervention_reference(monkeypatch):
    from lerobot.async_inference import policy_server_drtc
    from lerobot.transport import services_pb2

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)

    server = _server_for_accept_transition(policy_server_drtc)
    server._rlt_intervention_reference_mode = "original"

    source_reference = torch.ones(3, 2)
    source = policy_server_drtc.RLTSourceContext(
        context_id=1,
        source_control_step=10,
        chunk_start_step=10,
        rl_token=torch.ones(4),
        proprio=torch.ones(2),
        reference_chunk=source_reference,
        anchor_state=None,
    )
    server._rlt_context_cache.put(source)

    executed = torch.zeros(3, 2, dtype=torch.float32)
    transition = services_pb2.RLTTransitionChunk(
        episode_id=7,
        source_rlt_context_id=1,
        next_rlt_context_id=0,
        chunk_start_step=10,
        num_actions=3,
        action_dim=2,
        executed_actions_f32=executed.numpy().tobytes(),
        reward=0.0,
        done=True,
        is_intervention=True,
    )

    server._accept_rlt_transition(transition)

    sample = server._rlt_replay.samples()[0]
    assert torch.equal(sample.reference_chunk, source_reference)
    assert torch.equal(sample.executed_chunk, executed)


@require_package("grpcio", "grpc")
def test_accept_rlt_transition_keeps_source_reference_for_non_intervention(monkeypatch):
    from lerobot.async_inference import policy_server_drtc
    from lerobot.transport import services_pb2

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)

    server = _server_for_accept_transition(policy_server_drtc)

    source_reference = torch.ones(3, 2)
    source = policy_server_drtc.RLTSourceContext(
        context_id=1,
        source_control_step=10,
        chunk_start_step=10,
        rl_token=torch.ones(4),
        proprio=torch.ones(2),
        reference_chunk=source_reference,
        anchor_state=None,
    )
    server._rlt_context_cache.put(source)

    executed = torch.zeros(3, 2, dtype=torch.float32)
    transition = services_pb2.RLTTransitionChunk(
        episode_id=7,
        source_rlt_context_id=1,
        next_rlt_context_id=0,
        chunk_start_step=10,
        num_actions=3,
        action_dim=2,
        executed_actions_f32=executed.numpy().tobytes(),
        reward=0.0,
        done=True,
        is_intervention=False,
    )

    server._accept_rlt_transition(transition)

    sample = server._rlt_replay.samples()[0]
    assert torch.equal(sample.reference_chunk, source_reference)
    assert torch.equal(sample.executed_chunk, executed)
