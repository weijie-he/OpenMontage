"""Offline contract and request-shape tests for MiniMax video and TTS."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from tools.audio.minimax_tts import MiniMaxTTS
from tools.base_tool import ToolStatus
from tools.tool_registry import ToolRegistry
from tools.video.minimax_video import MiniMaxVideo


class _Response:
    def __init__(self, data=None, *, content: bytes = b"") -> None:
        self._data = data or {}
        self.content = content

    def json(self):  # noqa: ANN201 - mirrors requests.Response
        return self._data

    def raise_for_status(self) -> None:
        return None


def test_identity_and_shared_key_status():
    video = MiniMaxVideo()
    tts = MiniMaxTTS()

    assert video.get_info()["provider"] == "minimax"
    assert video.get_info()["capability"] == "video_generation"
    assert tts.get_info()["provider"] == "minimax"
    assert tts.get_info()["capability"] == "tts"
    assert "minimax" in video.agent_skills
    assert "minimax" in tts.agent_skills
    assert "text-to-speech" not in tts.agent_skills

    with patch.dict(os.environ, {}, clear=True):
        assert video.get_status() == ToolStatus.UNAVAILABLE
        assert tts.get_status() == ToolStatus.UNAVAILABLE

    with patch.dict(os.environ, {"FAL_KEY": "fal-test"}, clear=True):
        assert video.get_status() == ToolStatus.AVAILABLE
        assert tts.get_status() == ToolStatus.UNAVAILABLE

    with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}, clear=True):
        assert video.get_status() == ToolStatus.AVAILABLE
        assert tts.get_status() == ToolStatus.AVAILABLE


def test_registry_discovers_video_and_tts():
    registry = ToolRegistry()
    registry.discover()

    assert registry.get("minimax_video") is not None
    assert registry.get("minimax_tts") is not None
    assert {tool.name for tool in registry.get_by_provider("minimax")} >= {
        "minimax_video",
        "minimax_tts",
    }


def test_video_backend_auto_prefers_direct_then_fal():
    tool = MiniMaxVideo()
    with patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "mm", "FAL_KEY": "fal"},
        clear=True,
    ):
        assert tool._resolve_backend({"backend": "auto"}) == "direct"
        assert tool._resolve_backend({"backend": "minimax"}) == "direct"
        assert tool._resolve_backend({"backend": "fal"}) == "fal"

    with patch.dict(os.environ, {"FAL_KEY": "fal"}, clear=True):
        assert tool._resolve_backend({"backend": "auto"}) == "fal"

    with patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "new-direct-key", "FAL_KEY": "fal"},
        clear=True,
    ):
        assert tool._resolve_backend(
            {
                "backend": "auto",
                "fal_status_url": "https://queue.fal.run/status",
                "fal_response_url": "https://queue.fal.run/response",
            }
        ) == "fal"


def test_minimax_defaults_to_china_and_allows_explicit_global_host():
    with patch.dict(os.environ, {}, clear=True):
        assert MiniMaxVideo._direct_urls() == (
            "https://api.minimaxi.com/v1/video_generation",
            "https://api.minimaxi.com/v1/query/video_generation",
            "https://api.minimaxi.com/v1/files/retrieve",
        )
        assert MiniMaxTTS._endpoint() == "https://api.minimaxi.com/v1/t2a_v2"

    with patch.dict(
        os.environ,
        {"MINIMAX_API_BASE_URL": "https://api.minimax.io/"},
        clear=True,
    ):
        assert MiniMaxVideo._direct_urls()[0] == "https://api.minimax.io/v1/video_generation"
        assert MiniMaxTTS._endpoint() == "https://api.minimax.io/v1/t2a_v2"


def test_minimax_rejects_unofficial_api_host_before_forwarding_credentials():
    with patch.dict(
        os.environ,
        {"MINIMAX_API_BASE_URL": "https://example.com"},
        clear=True,
    ):
        with pytest.raises(ValueError, match="api.minimaxi.com"):
            MiniMaxVideo._direct_urls()
        with pytest.raises(ValueError, match="api.minimaxi.com"):
            MiniMaxTTS._endpoint()


def test_invalid_video_backend_never_falls_through_to_paid_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: pytest.fail("invalid backend must not submit"),
    )
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: pytest.fail("invalid backend must not poll"),
    )
    with patch.dict(os.environ, {"FAL_KEY": "fal-only"}, clear=True):
        result = MiniMaxVideo().execute(
            {
                "backend": "diret",
                "prompt": "river",
                "output_path": str(tmp_path / "clip.mp4"),
            }
        )

    assert result.success is False
    assert result.data["charge_state"] == "not_submitted"
    assert result.data["submission_attempted"] is False
    assert result.data["cost_estimate_status"] == "invalid_backend"
    assert "invalid" in (result.error or "")


def test_direct_video_t2v_submit_poll_retrieve_and_download(tmp_path, monkeypatch):
    output = tmp_path / "clip.mp4"
    post_calls = []
    get_calls = []

    def fake_post(url, **kwargs):  # noqa: ANN001, ANN202
        post_calls.append((url, kwargs))
        return _Response({"task_id": "task-1", "base_resp": {"status_code": 0}})

    def fake_get(url, **kwargs):  # noqa: ANN001, ANN202
        get_calls.append((url, kwargs))
        if url == MiniMaxVideo.DIRECT_QUERY_URL:
            return _Response(
                {
                    "task_id": "task-1",
                    "status": "Success",
                    "file_id": "file-1",
                    "video_width": 1920,
                    "video_height": 1080,
                    "base_resp": {"status_code": 0},
                }
            )
        if url == MiniMaxVideo.DIRECT_RETRIEVE_URL:
            return _Response(
                {
                    "file": {"download_url": "https://cdn.example/clip.mp4"},
                    "base_resp": {"status_code": 0},
                }
            )
        assert url == "https://cdn.example/clip.mp4"
        return _Response(content=b"\x00\x00\x00\x18ftypmp42fake-mp4")

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "secret-mm"}, clear=True):
        result = MiniMaxVideo().execute(
            {
                "backend": "direct",
                "operation": "text_to_video",
                "prompt": "A girl looks across the river [Static shot].",
                "duration": "6",
                "resolution": "1080P",
                "output_path": str(output),
            }
        )

    assert result.success is True
    assert output.read_bytes() == b"\x00\x00\x00\x18ftypmp42fake-mp4"
    assert result.data["task_id"] == "task-1"
    assert result.data["file_id"] == "file-1"
    assert result.data["backend"] == "direct"
    assert result.data["video_duration_seconds"] == 6
    assert post_calls[0][0] == MiniMaxVideo.DIRECT_CREATE_URL
    assert post_calls[0][1]["headers"]["Authorization"] == "Bearer secret-mm"
    assert post_calls[0][1]["json"]["model"] == "MiniMax-Hailuo-2.3"
    assert post_calls[0][1]["json"]["duration"] == 6
    assert get_calls[0][1]["params"] == {"task_id": "task-1"}
    assert get_calls[1][1]["params"] == {"file_id": "file-1"}


def test_direct_video_local_reference_becomes_data_url(tmp_path):
    image_path = tmp_path / "first.png"
    Image.new("RGB", (360, 640), color=(120, 160, 180)).save(image_path)

    payload = MiniMaxVideo()._build_direct_payload(
        {
            "operation": "image_to_video",
            "prompt": "She turns toward the drumbeat.",
            "reference_image_path": str(image_path),
            "duration": 6,
            "resolution": "1080P",
        }
    )

    assert payload["first_frame_image"].startswith("data:image/png;base64,")
    assert payload["model"] == "MiniMax-Hailuo-2.3"


def test_direct_video_rejects_unsupported_duration_resolution_pair():
    with pytest.raises(ValueError, match="1080P"):
        MiniMaxVideo()._build_direct_payload(
            {
                "operation": "text_to_video",
                "prompt": "river",
                "duration": 10,
                "resolution": "1080P",
            }
        )


def test_direct_video_validates_operation_specific_models_and_hailuo02_resolution():
    tool = MiniMaxVideo()
    with pytest.raises(ValueError, match="does not support text_to_video"):
        tool._build_direct_payload(
            {
                "operation": "text_to_video",
                "model": "MiniMax-Hailuo-2.3-Fast",
                "prompt": "river",
                "duration": 6,
                "resolution": "1080P",
            }
        )

    payload = tool._build_direct_payload(
        {
            "operation": "image_to_video",
            "model": "MiniMax-Hailuo-02",
            "prompt": "river",
            "duration": 10,
            "resolution": "512P",
            "image_url": "https://example.com/first.png",
        }
    )
    assert payload["resolution"] == "512P"


def test_direct_video_business_error_and_key_redaction():
    with pytest.raises(RuntimeError, match="1008"):
        MiniMaxVideo._raise_for_api_error(
            {"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}},
            "create task",
        )

    with patch.dict(os.environ, {"MINIMAX_API_KEY": "do-not-leak"}, clear=True):
        assert "do-not-leak" not in MiniMaxVideo._safe_error(RuntimeError("bad do-not-leak"))


def test_direct_video_timeout_returns_resume_id_without_blind_retry(tmp_path, monkeypatch):
    monotonic_values = iter([0.0, 31.0])
    monkeypatch.setattr(
        "tools.video.minimax_video.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: _Response(
            {"task_id": "task-resume", "base_resp": {"status_code": 0}}
        ),
    )
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: _Response(
            {"status": "Processing", "base_resp": {"status_code": 0}}
        ),
    )

    with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm"}, clear=True):
        result = MiniMaxVideo().execute(
            {
                "backend": "direct",
                "prompt": "river",
                "timeout_seconds": 30,
                "output_path": str(tmp_path / "clip.mp4"),
            }
        )

    assert result.success is False
    assert result.data["task_id"] == "task-resume"
    assert result.data["resume_inputs"]["task_id"] == "task-resume"
    assert result.data["submitted_this_call"] is True
    assert MiniMaxVideo().retry_policy.max_retries == 0


def test_direct_video_submit_timeout_marks_unknown_charge_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("read timed out")),
    )
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: pytest.fail("no task ID means polling must not start"),
    )

    with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm"}, clear=True):
        result = MiniMaxVideo().execute(
            {
                "backend": "direct",
                "prompt": "river",
                "output_path": str(tmp_path / "clip.mp4"),
            }
        )

    assert result.success is False
    assert result.data["submission_attempted"] is True
    assert result.data["submitted_this_call"] is False
    assert result.data["charge_state"] == "unknown"
    assert result.data["resubmit_safe"] is False
    assert result.data["potential_cost_usd"] == pytest.approx(0.49)
    assert result.cost_usd == 0.0
    assert "must not be resubmitted blindly" in (result.error or "")


def test_direct_video_resume_skips_paid_submit(tmp_path, monkeypatch):
    output = tmp_path / "resumed.mp4"
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: pytest.fail("resume must not submit a new task"),
    )

    def fake_get(url, **kwargs):  # noqa: ANN001, ANN202
        if url == MiniMaxVideo.DIRECT_QUERY_URL:
            return _Response(
                {"status": "Success", "file_id": "file-resume", "base_resp": {"status_code": 0}}
            )
        if url == MiniMaxVideo.DIRECT_RETRIEVE_URL:
            return _Response(
                {"file": {"download_url": "https://cdn.example/resumed.mp4"}, "base_resp": {"status_code": 0}}
            )
        return _Response(content=b"\x00\x00\x00\x18ftypmp42resumed-video")

    monkeypatch.setattr("requests.get", fake_get)
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm"}, clear=True):
        result = MiniMaxVideo().execute(
            {
                "backend": "direct",
                "task_id": "task-resume",
                "output_path": str(output),
            }
        )

    assert result.success is True
    assert result.data["submitted_this_call"] is False
    assert result.cost_usd == 0.0
    assert output.read_bytes() == b"\x00\x00\x00\x18ftypmp42resumed-video"


def test_fal_payload_and_cost_match_variant_contract():
    tool = MiniMaxVideo()
    payload = tool._build_fal_payload(
        {
            "prompt": "river",
            "duration": 10,
            "resolution": "768P",
            "image_url": "https://example.com/first.png",
        },
        operation="image_to_video",
        variant="hailuo-02/standard",
    )
    assert payload["duration"] == "10"
    assert payload["resolution"] == "768P"
    assert payload["prompt_optimizer"] is True
    assert tool.estimate_cost(
        {"backend": "fal", "model_variant": "hailuo-02/pro", "duration": 6}
    ) == pytest.approx(0.48)

    pro_payload = tool._build_fal_payload(
        {
            "prompt": "river",
            "image_url": "https://example.com/first.png",
        },
        operation="image_to_video",
        variant="hailuo-02/pro",
    )
    assert "duration" not in pro_payload
    assert "resolution" not in pro_payload

    t2v_payload = tool._build_fal_payload(
        {"prompt": "river", "duration": 6, "resolution": "768P"},
        operation="text_to_video",
        variant="hailuo-02/standard",
    )
    assert t2v_payload["duration"] == "6"
    assert "resolution" not in t2v_payload
    assert tool.estimate_cost(
        {
            "backend": "fal",
            "model_variant": "hailuo-02/standard",
            "operation": "text_to_video",
            "resolution": "512P",
            "duration": 6,
        }
    ) == pytest.approx(0.27)


def test_fal_partial_resume_urls_refuse_paid_resubmit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: pytest.fail("partial resume must not submit a paid task"),
    )
    with patch.dict(os.environ, {"FAL_KEY": "fal"}, clear=True):
        result = MiniMaxVideo().execute(
            {
                "backend": "fal",
                "prompt": "river",
                "fal_status_url": "https://queue.fal.run/status?id=signed",
                "output_path": str(tmp_path / "clip.mp4"),
            }
        )
    assert result.success is False
    assert "requires both" in (result.error or "")
    assert result.data["submitted_this_call"] is False


def test_cross_backend_resume_fields_never_submit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: pytest.fail("cross-backend resume must not submit"),
    )
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: pytest.fail("cross-backend resume must not poll"),
    )
    tool = MiniMaxVideo()

    with patch.dict(os.environ, {"FAL_KEY": "fal"}, clear=True):
        fal_result = tool.execute(
            {
                "backend": "fal",
                "task_id": "official-task",
                "prompt": "river",
                "output_path": str(tmp_path / "fal.mp4"),
            }
        )
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm"}, clear=True):
        direct_result = tool.execute(
            {
                "backend": "direct",
                "fal_status_url": "https://queue.fal.run/status",
                "fal_response_url": "https://queue.fal.run/response",
                "prompt": "river",
                "output_path": str(tmp_path / "direct.mp4"),
            }
        )

    assert fal_result.success is False
    assert direct_result.success is False
    assert "cannot consume" in (fal_result.error or "")
    assert "cannot consume" in (direct_result.error or "")


def test_resume_cost_estimates_are_zero_for_both_backends():
    tool = MiniMaxVideo()
    with patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "mm", "FAL_KEY": "fal"},
        clear=True,
    ):
        assert tool.estimate_cost(
            {"backend": "direct", "task_id": "task-existing"}
        ) == 0.0
        assert tool.estimate_cost(
            {
                "backend": "fal",
                "fal_status_url": "https://queue.fal.run/status",
                "fal_response_url": "https://queue.fal.run/response",
            }
        ) == 0.0


def test_fal_resume_rejects_untrusted_queue_urls_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: pytest.fail("resume URL validation must precede POST"),
    )
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: pytest.fail("untrusted URL must never receive FAL_KEY"),
    )
    with patch.dict(os.environ, {"FAL_KEY": "do-not-leak"}, clear=True):
        result = MiniMaxVideo().execute(
            {
                "backend": "fal",
                "fal_status_url": "https://attacker.example/status",
                "fal_response_url": "https://attacker.example/response",
                "output_path": str(tmp_path / "clip.mp4"),
            }
        )

    assert result.success is False
    assert "approved fal.ai queue host" in (result.error or "")
    assert "do-not-leak" not in (result.error or "")
    assert result.data["submitted_this_call"] is False
    assert result.cost_usd == 0.0


def test_fal_submit_timeout_marks_unknown_charge_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("read timed out")),
    )
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: pytest.fail("missing queue URLs must prevent polling"),
    )
    with patch.dict(os.environ, {"FAL_KEY": "fal"}, clear=True):
        result = MiniMaxVideo().execute(
            {
                "backend": "fal",
                "prompt": "river",
                "model_variant": "hailuo-02/pro",
                "output_path": str(tmp_path / "clip.mp4"),
            }
        )

    assert result.success is False
    assert result.data["submission_attempted"] is True
    assert result.data["submitted_this_call"] is False
    assert result.data["charge_state"] == "unknown"
    assert result.data["resubmit_safe"] is False
    assert result.data["potential_cost_usd"] == pytest.approx(0.48)
    assert result.cost_usd == 0.0
    assert "must not be resubmitted blindly" in (result.error or "")


def test_fal_submit_response_rejects_untrusted_queue_urls_before_authenticated_get(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: _Response(
            {
                "request_id": "request-created",
                "status_url": "https://attacker.example/status",
                "response_url": "https://attacker.example/response",
            }
        ),
    )
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: pytest.fail("untrusted provider URL must not receive FAL_KEY"),
    )
    with patch.dict(os.environ, {"FAL_KEY": "do-not-leak"}, clear=True):
        result = MiniMaxVideo().execute(
            {
                "backend": "fal",
                "prompt": "river",
                "model_variant": "hailuo-02/pro",
                "output_path": str(tmp_path / "clip.mp4"),
            }
        )

    assert result.success is False
    assert result.data["submission_attempted"] is True
    assert result.data["submitted_this_call"] is True
    assert result.data["request_id"] == "request-created"
    assert result.cost_usd == pytest.approx(0.48)
    assert "approved fal.ai queue host" in (result.error or "")
    assert "do-not-leak" not in (result.error or "")


def test_tts_payload_maps_selector_aliases():
    payload = MiniMaxTTS()._build_payload(
        {
            "text": "端午又快到了。",
            "model_id": "speech-2.8-hd",
            "voice_id": "male-qn-qingse",
            "speaking_rate": 0.9,
            "pitch": -1,
            "language_boost": "Chinese",
            "output_format": "mp3_44100_128",
            "pronunciation_tone": ["茶峒/(cha2)(dong4)"],
            "subtitle_enable": True,
            "subtitle_type": "word",
        }
    )

    assert payload["model"] == "speech-2.8-hd"
    assert payload["stream"] is False
    assert payload["output_format"] == "hex"
    assert payload["voice_setting"]["voice_id"] == "male-qn-qingse"
    assert payload["voice_setting"]["speed"] == pytest.approx(0.9)
    assert payload["voice_setting"]["pitch"] == -1
    assert payload["audio_setting"]["format"] == "mp3"
    assert payload["audio_setting"]["sample_rate"] == 44100
    assert payload["audio_setting"]["bitrate"] == 128000
    assert payload["pronunciation_dict"]["tone"] == ["茶峒/(cha2)(dong4)"]
    assert payload["subtitle_enable"] is True
    assert payload["subtitle_type"] == "word"


def test_tts_maps_cantonese_alias_and_validates_model_specific_emotion():
    payload = MiniMaxTTS()._build_payload(
        {"text": "早晨", "language_boost": "Cantonese"}
    )
    assert payload["language_boost"] == "Chinese,Yue"

    with pytest.raises(ValueError, match="speech-2.6"):
        MiniMaxTTS()._build_payload(
            {"text": "test", "model": "speech-2.8-hd", "emotion": "whisper"}
        )


def test_tts_decodes_hex_audio_and_returns_usage(tmp_path, monkeypatch):
    output = tmp_path / "line.mp3"
    calls = []

    def fake_post(url, **kwargs):  # noqa: ANN001, ANN202
        calls.append((url, kwargs))
        return _Response(
            {
                "data": {"audio": b"ID3-minimax".hex(), "status": 2},
                "extra_info": {
                    "audio_length": 1234,
                    "audio_size": 11,
                    "word_count": 4,
                    "usage_characters": 8,
                },
                "trace_id": "trace-1",
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        )

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(MiniMaxTTS, "_probe_duration", staticmethod(lambda path: 1.234))
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "secret-mm"}, clear=True):
        result = MiniMaxTTS().execute(
            {
                "text": "端午又快到了。",
                "voice_id": "male-qn-qingse",
                "output_path": str(output),
            }
        )

    assert result.success is True
    assert output.read_bytes() == b"ID3-minimax"
    assert result.data["audio_duration_seconds"] == pytest.approx(1.23)
    assert result.data["usage"]["usage_characters"] == 8
    assert calls[0][0] == MiniMaxTTS.DEFAULT_ENDPOINT
    assert calls[0][1]["headers"]["Authorization"] == "Bearer secret-mm"
    assert calls[0][1]["json"]["output_format"] == "hex"


def test_tts_uses_fallback_key_only_after_explicit_usage_limit(tmp_path, monkeypatch):
    calls = []

    def fake_post(url, **kwargs):  # noqa: ANN001, ANN202
        calls.append(kwargs["headers"]["Authorization"])
        if len(calls) == 1:
            return _Response(
                {"base_resp": {"status_code": 2056, "status_msg": "usage limit reached"}}
            )
        return _Response(
            {
                "data": {"audio": b"ID3-fallback".hex(), "status": 2},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        )

    monkeypatch.setattr("requests.post", fake_post)
    with patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "primary", "MINIMAX_API_KEY_FALLBACK": "fallback"},
        clear=True,
    ):
        result = MiniMaxTTS().execute(
            {"text": "test", "output_path": str(tmp_path / "fallback.mp3")}
        )

    assert result.success is True
    assert calls == ["Bearer primary", "Bearer fallback"]
    assert result.data["credential_slot"] == "fallback"
    assert result.data["credential_attempts"] == 2


def test_tts_does_not_rotate_credentials_after_unknown_failure(tmp_path, monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append(kwargs["headers"]["Authorization"])
        raise TimeoutError("read timed out")

    monkeypatch.setattr("requests.post", fake_post)
    with patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "primary", "MINIMAX_API_KEY_FALLBACK": "fallback"},
        clear=True,
    ):
        result = MiniMaxTTS().execute(
            {"text": "test", "output_path": str(tmp_path / "timeout.mp3")}
        )

    assert result.success is False
    assert calls == ["Bearer primary"]
    assert result.data["credential_attempts"] == 1
    assert result.data["charge_state"] == "unknown"


def test_tts_business_error_invalid_hex_and_key_redaction(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="1004"):
        MiniMaxTTS._raise_for_api_error(
            {"base_resp": {"status_code": 1004, "status_msg": "auth failed"}}
        )

    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: _Response(
            {"data": {"audio": "not-hex", "status": 2}, "base_resp": {"status_code": 0}}
        ),
    )
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "do-not-leak"}, clear=True):
        result = MiniMaxTTS().execute(
            {"text": "test", "output_path": str(tmp_path / "bad.mp3")}
        )
        safe = MiniMaxTTS._safe_error(RuntimeError("bad do-not-leak"))

    assert result.success is False
    assert "hexadecimal" in (result.error or "")
    assert "do-not-leak" not in safe


def test_tts_completed_then_local_write_failure_keeps_cost_and_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: _Response(
            {
                "data": {"audio": b"ID3-paid".hex(), "status": 2},
                "extra_info": {"usage_characters": 4, "audio_length": 600},
                "trace_id": "trace-paid",
                "base_resp": {"status_code": 0},
            }
        ),
    )
    monkeypatch.setattr(
        Path,
        "write_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    inputs = {"text": "一二三四", "output_path": str(tmp_path / "line.mp3")}
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm"}, clear=True):
        result = MiniMaxTTS().execute(inputs)

    assert result.success is False
    assert result.data["synthesis_completed"] is True
    assert result.data["charge_state"] == "completed_estimated"
    assert result.data["resubmit_safe"] is False
    assert result.data["trace_id"] == "trace-paid"
    assert result.data["usage"]["usage_characters"] == 4
    assert result.cost_usd == pytest.approx(MiniMaxTTS().estimate_cost(inputs))
    assert "do not resubmit blindly" in (result.error or "")


def test_tts_post_timeout_marks_unknown_potential_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("read timed out")),
    )
    inputs = {"text": "一二三四", "output_path": str(tmp_path / "line.mp3")}
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm"}, clear=True):
        result = MiniMaxTTS().execute(inputs)

    assert result.success is False
    assert result.data["request_attempted"] is True
    assert result.data["synthesis_completed"] is False
    assert result.data["charge_state"] == "unknown"
    assert result.data["resubmit_safe"] is False
    assert result.data["potential_cost_usd"] == pytest.approx(
        MiniMaxTTS().estimate_cost(inputs)
    )
    assert result.cost_usd == 0.0
    assert "do not resubmit blindly" in (result.error or "")


def test_tts_rejects_incomplete_non_streaming_status(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: _Response(
            {
                "data": {"audio": b"ID3-partial".hex(), "status": 1},
                "base_resp": {"status_code": 0},
            }
        ),
    )
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm"}, clear=True):
        result = MiniMaxTTS().execute(
            {"text": "test", "output_path": str(tmp_path / "partial.mp3")}
        )
    assert result.success is False
    assert "incomplete" in (result.error or "")


def test_safe_errors_redact_signed_url_query_values():
    message = "download failed https://cdn.example/a.mp4?token=secret-token&sig=secret-signature"
    assert "secret-token" not in MiniMaxVideo._safe_error(RuntimeError(message))
    assert "secret-signature" not in MiniMaxTTS._safe_error(RuntimeError(message))


def test_direct_cost_uses_published_paygo_rates_and_allows_overrides():
    tool = MiniMaxVideo()
    tts = MiniMaxTTS()
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm"}, clear=True):
        assert tool.estimate_cost(
            {"backend": "direct", "model": "MiniMax-Hailuo-2.3", "resolution": "1080P", "duration": 6}
        ) == pytest.approx(0.49)
        assert tool.estimate_cost(
            {
                "backend": "direct",
                "operation": "image_to_video",
                "model": "MiniMax-Hailuo-2.3-Fast",
                "resolution": "768P",
                "duration": 6,
            }
        ) == pytest.approx(0.19)
        assert tts.estimate_cost({"text": "一二三四", "model": "speech-2.8-hd"}) == pytest.approx(0.0004)

    with patch.dict(
        os.environ,
        {
            "MINIMAX_API_KEY": "mm",
            "MINIMAX_VIDEO_COST_PER_SECOND_USD": "0.05",
            "MINIMAX_TTS_COST_PER_1000_CHARS_USD": "0.10",
        },
        clear=True,
    ):
        assert tool.estimate_cost({"backend": "direct", "duration": 6}) == pytest.approx(0.30)
        assert tts.estimate_cost({"text": "字" * 1000}) == pytest.approx(0.10)

    with patch.dict(
        os.environ,
        {
            "MINIMAX_API_KEY": "mm",
            "MINIMAX_VIDEO_COST_PER_SECOND_USD": "-1",
            "MINIMAX_TTS_COST_PER_1000_CHARS_USD": "not-a-number",
        },
        clear=True,
    ):
        assert tool.estimate_cost({"backend": "direct", "duration": 6}) == pytest.approx(0.49)
        assert tts.estimate_cost({"text": "字" * 1000}) == pytest.approx(0.10)


def test_tts_rejects_text_at_api_limit():
    with pytest.raises(ValueError, match="10,000"):
        MiniMaxTTS()._build_payload({"text": "字" * 10000})
