"""MiniMax Speech 2.8 text-to-speech via the official HTTP API."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class MiniMaxTTS(BaseTool):
    name = "minimax_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "minimax"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set MINIMAX_API_KEY to a MiniMax Platform API key.\n"
        "  Get one at https://platform.minimax.io/user-center/basic-information/interface-key"
    )
    fallback = "piper_tts"
    fallback_tools = [
        "dashscope_tts",
        "doubao_tts",
        "elevenlabs_tts",
        "openai_tts",
        "piper_tts",
    ]
    agent_skills = ["minimax"]

    capabilities = ["text_to_speech", "voice_selection", "multilingual"]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
        "speed_control": True,
        "pitch_control": True,
        "pronunciation_dictionary": True,
        "subtitle_timestamps": True,
        "pause_tags": True,
        "published_paygo_cost_estimate": True,
    }
    best_for = [
        "natural Mandarin narration and character dialogue",
        "using one MiniMax credential for both Hailuo video and speech",
        "voice delivery with speed, volume, pitch, and emotion controls",
    ]
    not_good_for = ["offline production", "requests of 10,000 characters or more"]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {
                "type": "string",
                "maxLength": 9999,
                "description": "Text to synthesize. Must be shorter than 10,000 characters.",
            },
            "model": {
                "type": "string",
                "enum": [
                    "speech-2.8-hd",
                    "speech-2.8-turbo",
                    "speech-2.6-hd",
                    "speech-2.6-turbo",
                    "speech-02-hd",
                    "speech-02-turbo",
                ],
                "default": "speech-2.8-hd",
            },
            "model_id": {
                "type": "string",
                "description": "Selector-compatible alias for model.",
            },
            "voice_id": {
                "type": "string",
                "default": "male-qn-qingse",
                "description": "MiniMax system or account voice ID.",
            },
            "voice": {
                "type": "string",
                "description": "Alias for voice_id.",
            },
            "speed": {"type": "number", "minimum": 0.5, "maximum": 2.0, "default": 1.0},
            "speaking_rate": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 2.0,
                "description": "Selector-compatible alias for speed.",
            },
            "volume": {"type": "number", "exclusiveMinimum": 0, "maximum": 10, "default": 1.0},
            "pitch": {"type": "integer", "minimum": -12, "maximum": 12, "default": 0},
            "emotion": {
                "type": "string",
                "enum": ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm", "fluent", "whisper"],
                "description": "Optional delivery emotion when supported by the selected voice/model.",
            },
            "language_boost": {
                "type": "string",
                "default": "auto",
                "description": "Language hint such as auto, Chinese, English, Japanese, or Cantonese.",
            },
            "format": {
                "type": "string",
                "enum": ["mp3", "wav", "pcm", "flac"],
                "default": "mp3",
            },
            "output_format": {
                "type": "string",
                "description": (
                    "Selector-compatible format alias, e.g. mp3 or mp3_44100_128. "
                    "An encoded sample rate/bitrate is used unless the explicit fields override it."
                ),
            },
            "sample_rate": {
                "type": "integer",
                "enum": [8000, 16000, 22050, 24000, 32000, 44100],
                "default": 32000,
            },
            "bitrate": {
                "type": "integer",
                "enum": [32000, 64000, 128000, 256000],
                "default": 128000,
            },
            "channel": {"type": "integer", "enum": [1, 2], "default": 1},
            "pronunciation_tone": {
                "type": "array",
                "items": {"type": "string"},
                "description": "MiniMax pronunciation entries, e.g. '处理/(chu3)(li3)'.",
            },
            "voice_modify": {
                "type": "object",
                "description": "Optional MiniMax voice_modify object passed through unchanged.",
            },
            "subtitle_enable": {
                "type": "boolean",
                "default": False,
                "description": "Ask MiniMax to return subtitle timestamps.",
            },
            "subtitle_type": {
                "type": "string",
                "enum": ["sentence", "word"],
                "default": "sentence",
            },
            "output_path": {"type": "string"},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "output": {"type": "string"},
            "audio_duration_seconds": {"type": ["number", "null"]},
            "usage": {"type": "object"},
            "subtitle_data": {"type": ["object", "array", "string", "null"]},
            "trace_id": {"type": ["string", "null"]},
            "synthesis_completed": {"type": "boolean"},
            "request_attempted": {"type": "boolean"},
            "charge_state": {"type": "string"},
            "resubmit_safe": {"type": "boolean"},
            "potential_cost_usd": {"type": "number"},
        },
    }
    artifact_schema = {"type": "array", "items": {"type": "string"}}

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(
        # A synchronous request may finish server-side after the client times
        # out, and it cannot be resumed from trace_id. Never blind-retry a paid
        # synthesis call; callers must explicitly decide whether to resubmit.
        max_retries=0,
        backoff_seconds=0.0,
        retryable_errors=[],
    )
    idempotency_key_fields = [
        "text",
        "model",
        "model_id",
        "voice_id",
        "voice",
        "speed",
        "speaking_rate",
        "pitch",
        "emotion",
        "format",
        "subtitle_enable",
        "subtitle_type",
    ]
    side_effects = [
        "writes audio file to output_path",
        "calls the MiniMax text-to-audio API",
    ]
    user_visible_verification = [
        "Listen to an approval sample for Mandarin naturalness, character fit, pacing, and pronunciation",
        "When subtitles are requested, verify returned word/sentence timing before composition",
    ]

    DEFAULT_ENDPOINT = "https://api.minimax.io/v1/t2a_v2"
    PUBLISHED_PAYGO_RATE_PER_CHARACTER_USD = {
        "speech-2.8-hd": 0.0001,
        "speech-2.8-turbo": 0.00006,
        "speech-2.6-hd": 0.0001,
        "speech-2.6-turbo": 0.00006,
        "speech-02-hd": 0.0001,
        "speech-02-turbo": 0.00006,
    }
    LANGUAGE_ALIASES = {
        "Cantonese": "Chinese,Yue",
        "Yue": "Chinese,Yue",
        "zh-yue": "Chinese,Yue",
    }

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if os.environ.get("MINIMAX_API_KEY") else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        rate = self._override_rate_per_1000_chars()
        if rate is not None:
            return round(len(str(inputs.get("text", ""))) / 1000 * rate, 4)

        model = str(inputs.get("model") or inputs.get("model_id") or "speech-2.8-hd")
        if model not in self.PUBLISHED_PAYGO_RATE_PER_CHARACTER_USD:
            raise ValueError(
                f"MiniMax TTS model {model} has no configured cost contract; use a current 2.8, 2.6, or 02 model"
            )
        published_rate = self.PUBLISHED_PAYGO_RATE_PER_CHARACTER_USD.get(model)
        if published_rate is None:
            return 0.0
        return round(len(str(inputs.get("text", ""))) * published_rate, 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        # A planning estimate only; the API returns actual audio metadata.
        return max(3.0, len(str(inputs.get("text", ""))) / 12)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            return ToolResult(success=False, error="MINIMAX_API_KEY not set. " + self.install_instructions)

        import requests

        started = time.time()
        payload: dict[str, Any] | None = None
        response_data: dict[str, Any] = {}
        response_payload: dict[str, Any] = {}
        synthesis_status: Any = None
        synthesis_completed = False
        request_attempted = False
        estimated_cost = 0.0
        try:
            payload = self._build_payload(inputs)
            estimated_cost = self.estimate_cost(inputs)
            endpoint = os.environ.get("MINIMAX_TTS_ENDPOINT", self.DEFAULT_ENDPOINT)
            request_attempted = True
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(10, 120),
            )
            response.raise_for_status()
            response_data = response.json()
            self._raise_for_api_error(response_data)

            response_payload = response_data.get("data") or {}
            synthesis_status = response_payload.get("status")
            if synthesis_status not in (2, "2"):
                raise RuntimeError(
                    f"MiniMax TTS returned incomplete non-streaming status={synthesis_status}"
                )
            # From this point the provider has explicitly confirmed that the
            # paid synthesis completed. Preserve that fact even if decoding,
            # media validation, or the requested local write later fails.
            synthesis_completed = True
            audio_hex = response_payload.get("audio")
            if not audio_hex:
                raise RuntimeError("MiniMax TTS response did not include data.audio")
            try:
                audio_bytes = bytes.fromhex(str(audio_hex))
            except ValueError as exc:
                raise RuntimeError("MiniMax TTS returned invalid hexadecimal audio data") from exc

            audio_format = payload["audio_setting"]["format"]
            self._validate_audio_bytes(audio_bytes, audio_format)
            output_path = Path(inputs.get("output_path", f"minimax_tts.{audio_format}"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)
            audio_duration = self._probe_duration(output_path)
        except Exception as exc:
            extra_info = response_data.get("extra_info") or {}
            usage = {
                "usage_characters": extra_info.get("usage_characters"),
                "word_count": extra_info.get("word_count"),
                "audio_size": extra_info.get("audio_size"),
                "audio_length_ms": extra_info.get("audio_length"),
            }
            if synthesis_completed:
                charge_state = "completed_estimated"
                retry_warning = (
                    "MiniMax confirmed synthesis status=2, so this request may have been charged; "
                    "do not resubmit blindly."
                )
            elif request_attempted:
                charge_state = "unknown"
                retry_warning = (
                    "The request was attempted but completion and charge could not be confirmed; "
                    "do not resubmit blindly."
                )
            else:
                charge_state = "not_submitted"
                retry_warning = "The request was not submitted."

            failure_data: dict[str, Any] = {
                "provider": self.provider,
                "synthesis_completed": synthesis_completed,
                "request_attempted": request_attempted,
                "charge_state": charge_state,
                "resubmit_safe": not request_attempted,
                "estimated_cost_usd": estimated_cost,
                "potential_cost_usd": estimated_cost if request_attempted else 0.0,
                "status": synthesis_status,
                "usage": usage,
                "trace_id": response_data.get("trace_id"),
            }
            if payload is not None:
                failure_data.update(
                    {
                        "model": payload["model"],
                        "voice_id": payload["voice_setting"]["voice_id"],
                        "format": payload["audio_setting"]["format"],
                        "sample_rate": payload["audio_setting"]["sample_rate"],
                        "cost_estimate_status": self._cost_estimate_status(inputs),
                    }
                )
            return ToolResult(
                success=False,
                data=failure_data,
                error=f"MiniMax TTS failed: {self._safe_error(exc)} {retry_warning}",
                cost_usd=estimated_cost if synthesis_completed else 0.0,
                duration_seconds=round(time.time() - started, 2),
                model=payload["model"] if payload is not None else None,
            )

        extra_info = response_data.get("extra_info") or {}
        if audio_duration is None and extra_info.get("audio_length") is not None:
            try:
                audio_duration = float(extra_info["audio_length"]) / 1000
            except (TypeError, ValueError):
                audio_duration = None
        usage = {
            "usage_characters": extra_info.get("usage_characters"),
            "word_count": extra_info.get("word_count"),
            "audio_size": extra_info.get("audio_size"),
            "audio_length_ms": extra_info.get("audio_length"),
        }
        subtitle_data = {
            key: value
            for key, value in response_payload.items()
            if key not in {"audio", "status"}
        } or None
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": payload["model"],
                "voice_id": payload["voice_setting"]["voice_id"],
                "format": payload["audio_setting"]["format"],
                "sample_rate": payload["audio_setting"]["sample_rate"],
                "text_length": len(str(inputs["text"])),
                "audio_duration_seconds": round(audio_duration, 2) if audio_duration else None,
                "output": str(output_path),
                "usage": usage,
                "subtitle_data": subtitle_data,
                "trace_id": response_data.get("trace_id"),
                "synthesis_completed": True,
                "request_attempted": True,
                "charge_state": "completed_estimated",
                "resubmit_safe": False,
                "potential_cost_usd": estimated_cost,
                "cost_estimate_status": self._cost_estimate_status(inputs),
            },
            artifacts=[str(output_path)],
            cost_usd=estimated_cost,
            duration_seconds=round(time.time() - started, 2),
            model=payload["model"],
        )

    def _build_payload(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = str(inputs["text"])
        if len(text) >= 10000:
            raise ValueError("MiniMax TTS text must be shorter than 10,000 characters")
        model = str(inputs.get("model") or inputs.get("model_id") or "speech-2.8-hd")
        if model not in self.PUBLISHED_PAYGO_RATE_PER_CHARACTER_USD:
            raise ValueError(
                f"MiniMax TTS model {model} has no configured cost contract; use a current 2.8, 2.6, or 02 model"
            )
        voice_id = str(inputs.get("voice_id") or inputs.get("voice") or "male-qn-qingse")
        speed = float(inputs.get("speed", inputs.get("speaking_rate", 1.0)))
        volume = float(inputs.get("volume", inputs.get("vol", 1.0)))
        pitch = int(inputs.get("pitch", 0))
        audio_format, alias_sample_rate, alias_bitrate = self._parse_output_format(
            inputs.get("format") or inputs.get("output_format") or "mp3"
        )
        sample_rate = int(inputs.get("sample_rate", alias_sample_rate or 32000))
        bitrate = int(inputs.get("bitrate", alias_bitrate or 128000))
        channel = int(inputs.get("channel", 1))

        if not 0.5 <= speed <= 2.0:
            raise ValueError("MiniMax TTS speed must be between 0.5 and 2.0")
        if not 0 < volume <= 10:
            raise ValueError("MiniMax TTS volume must be greater than 0 and at most 10")
        if not -12 <= pitch <= 12:
            raise ValueError("MiniMax TTS pitch must be between -12 and 12")
        if sample_rate not in {8000, 16000, 22050, 24000, 32000, 44100}:
            raise ValueError("MiniMax TTS sample_rate is unsupported")
        if bitrate not in {32000, 64000, 128000, 256000}:
            raise ValueError("MiniMax TTS bitrate is unsupported")
        if channel not in {1, 2}:
            raise ValueError("MiniMax TTS channel must be 1 or 2")

        voice_setting: dict[str, Any] = {
            "voice_id": voice_id,
            "speed": speed,
            "vol": volume,
            "pitch": pitch,
        }
        if inputs.get("emotion"):
            if inputs["emotion"] in {"fluent", "whisper"} and model not in {
                "speech-2.6-hd",
                "speech-2.6-turbo",
            }:
                raise ValueError(
                    f"MiniMax emotion={inputs['emotion']} is supported only by speech-2.6 models"
                )
            voice_setting["emotion"] = inputs["emotion"]

        language_boost = str(inputs.get("language_boost", "auto"))
        language_boost = self.LANGUAGE_ALIASES.get(language_boost, language_boost)

        payload: dict[str, Any] = {
            "model": model,
            "text": text,
            "stream": False,
            "language_boost": language_boost,
            "output_format": "hex",
            "voice_setting": voice_setting,
            "audio_setting": {
                "sample_rate": sample_rate,
                "bitrate": bitrate,
                "format": audio_format,
                "channel": channel,
            },
        }
        if inputs.get("pronunciation_tone"):
            payload["pronunciation_dict"] = {"tone": list(inputs["pronunciation_tone"])}
        if inputs.get("voice_modify"):
            payload["voice_modify"] = dict(inputs["voice_modify"])
        if inputs.get("subtitle_enable"):
            payload["subtitle_enable"] = True
            payload["subtitle_type"] = inputs.get("subtitle_type", "sentence")
        return payload

    def _cost_estimate_status(self, inputs: dict[str, Any]) -> str:
        if self._override_rate_per_1000_chars() is not None:
            return "configured_override"
        model = str(inputs.get("model") or inputs.get("model_id") or "speech-2.8-hd")
        if model in self.PUBLISHED_PAYGO_RATE_PER_CHARACTER_USD:
            return "published_paygo"
        return "unconfigured"

    @staticmethod
    def _override_rate_per_1000_chars() -> float | None:
        raw = os.environ.get("MINIMAX_TTS_COST_PER_1000_CHARS_USD")
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    @staticmethod
    def _normalize_format(value: Any) -> str:
        return MiniMaxTTS._parse_output_format(value)[0]

    @staticmethod
    def _parse_output_format(value: Any) -> tuple[str, int | None, int | None]:
        normalized = str(value).lower()
        parts = normalized.split("_")
        audio_format = parts[0]
        if audio_format not in {"mp3", "wav", "pcm", "flac"}:
            raise ValueError("MiniMax TTS format must be mp3, wav, pcm, or flac")
        if len(parts) == 1:
            return audio_format, None, None
        if len(parts) not in {2, 3} or not all(part.isdigit() for part in parts[1:]):
            raise ValueError(
                "MiniMax TTS output_format must use format_sampleRate_bitrate, e.g. mp3_44100_128"
            )

        sample_rate = int(parts[1])
        bitrate = int(parts[2]) if len(parts) == 3 else None
        if bitrate is not None and bitrate <= 1000:
            bitrate *= 1000
        return audio_format, sample_rate, bitrate

    @staticmethod
    def _raise_for_api_error(data: dict[str, Any]) -> None:
        base_resp = data.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0, "0"):
            status_msg = base_resp.get("status_msg") or "unknown API error"
            raise RuntimeError(f"MiniMax API error {status_code}: {status_msg}")

    @staticmethod
    def _probe_duration(path: Path) -> float | None:
        try:
            from tools.analysis.audio_probe import probe_duration

            return probe_duration(path)
        except Exception:
            return None

    @staticmethod
    def _validate_audio_bytes(content: bytes, audio_format: str) -> None:
        if not content:
            raise RuntimeError("MiniMax TTS returned empty audio data")
        if audio_format == "mp3":
            has_id3 = content.startswith(b"ID3")
            has_frame_sync = (
                len(content) >= 2
                and content[0] == 0xFF
                and content[1] & 0xE0 == 0xE0
            )
            if not (has_id3 or has_frame_sync):
                raise RuntimeError("MiniMax TTS returned invalid MP3 bytes")
        elif audio_format == "wav" and not content.startswith(b"RIFF"):
            raise RuntimeError("MiniMax TTS returned invalid WAV bytes")
        elif audio_format == "flac" and not content.startswith(b"fLaC"):
            raise RuntimeError("MiniMax TTS returned invalid FLAC bytes")

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        message = str(exc)
        key = os.environ.get("MINIMAX_API_KEY")
        if key:
            message = message.replace(key, "[redacted]")
        message = re.sub(
            r"([?&][^=\s&]+)=([^&\s]+)",
            r"\1=[redacted]",
            message,
        )
        return message
