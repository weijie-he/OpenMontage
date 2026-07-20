"""Xiaomi MiMo-V2.5-TTS via the official OpenAI-compatible HTTP API."""

from __future__ import annotations

import base64
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


class MimoTTS(BaseTool):
    """Generate speech with Xiaomi's MiMo-V2.5-TTS model."""

    name = "mimo_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "xiaomi_mimo"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:MIMO_API_KEY"]
    install_instructions = (
        "Set MIMO_API_KEY to a key issued by Xiaomi MiMo at https://mimo.mi.com/."
    )
    agent_skills = ["mimo-v2.5-tts"]

    capabilities = ["text_to_speech", "voice_selection", "delivery_instructions", "mandarin"]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
        "delivery_instructions": True,
        "streaming": False,
        "limited_time_free": True,
    }
    best_for = [
        "natural Mandarin narration",
        "instruction-directed technology launch voiceovers",
        "MiMo-V2.5-TTS built-in Chinese voices",
    ]
    not_good_for = ["offline production", "custom voice cloning"]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "model": {
                "type": "string",
                "enum": ["mimo-v2.5-tts"],
                "default": "mimo-v2.5-tts",
            },
            "model_id": {"type": "string", "description": "Selector-compatible model alias."},
            "voice": {
                "type": "string",
                "enum": ["冰糖", "茉莉", "苏打", "白桦", "mimo_default"],
                "default": "白桦",
            },
            "voice_id": {"type": "string", "description": "Selector-compatible voice alias."},
            "instructions": {
                "type": "string",
                "description": "Natural-language performance direction sent as the user message.",
            },
            "format": {"type": "string", "enum": ["wav"], "default": "wav"},
            "output_format": {"type": "string", "enum": ["wav"]},
            "output_path": {"type": "string"},
            "timeout_seconds": {"type": "integer", "minimum": 10, "maximum": 300, "default": 120},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "output": {"type": "string"},
            "audio_duration_seconds": {"type": ["number", "null"]},
            "provider": {"type": "string"},
            "model": {"type": "string"},
            "voice": {"type": "string"},
            "charge_state": {"type": "string"},
            "cost_estimate_status": {"type": "string"},
        },
    }
    artifact_schema = {"type": "array", "items": {"type": "string"}}

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=0, backoff_seconds=0.0, retryable_errors=[])
    idempotency_key_fields = ["text", "model", "model_id", "voice", "voice_id", "instructions", "format"]
    side_effects = [
        "calls the Xiaomi MiMo chat completions API",
        "writes a WAV file to output_path",
    ]
    user_visible_verification = [
        "Listen to an approval sample for Mandarin pronunciation, pacing, and launch-film tone"
    ]

    ENDPOINT = "https://api.xiaomimimo.com/v1/chat/completions"
    MODEL = "mimo-v2.5-tts"
    VOICES = {"冰糖", "茉莉", "苏打", "白桦", "mimo_default"}

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if os.environ.get("MIMO_API_KEY", "").strip() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Official model page currently marks MiMo-V2.5-TTS as limited-time free.
        # Keep this explicit so a future pricing change is not mistaken for PAYG.
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return max(4.0, len(str(inputs.get("text", ""))) / 9)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("MIMO_API_KEY", "").strip()
        if not api_key:
            return ToolResult(success=False, error="MIMO_API_KEY is not set. " + self.install_instructions)

        started = time.time()
        request_attempted = False
        model: str | None = None
        voice: str | None = None
        try:
            payload, model, voice, audio_format = self._build_payload(inputs)
            timeout_seconds = int(inputs.get("timeout_seconds", 120))
            if not 10 <= timeout_seconds <= 300:
                raise ValueError("MiMo TTS timeout_seconds must be between 10 and 300")

            import requests

            request_attempted = True
            response = requests.post(
                self.ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(10, timeout_seconds),
            )
            if not response.ok:
                raise RuntimeError(self._http_error(response))
            response_data = response.json()
            audio_data = self._extract_audio_data(response_data)
            audio_bytes = self._decode_audio(audio_data)
            self._validate_wav(audio_bytes)

            output_path = Path(inputs.get("output_path") or "mimo_tts.wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)
            audio_duration = self._probe_duration(output_path)
        except Exception as exc:
            charge_state = "unknown_limited_time_free" if request_attempted else "not_submitted"
            return ToolResult(
                success=False,
                data={
                    "provider": self.provider,
                    "model": model,
                    "voice": voice,
                    "request_attempted": request_attempted,
                    "charge_state": charge_state,
                    "cost_estimate_status": "official_limited_time_free",
                },
                error=f"MiMo TTS failed: {self._safe_error(exc, api_key)}",
                cost_usd=0.0,
                duration_seconds=round(time.time() - started, 2),
                model=model,
            )

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model,
                "voice": voice,
                "format": audio_format,
                "text_length": len(str(inputs["text"])),
                "audio_duration_seconds": round(audio_duration, 2) if audio_duration else None,
                "output": str(output_path),
                "request_attempted": True,
                "charge_state": "completed_limited_time_free",
                "cost_estimate_status": "official_limited_time_free",
            },
            artifacts=[str(output_path)],
            cost_usd=0.0,
            duration_seconds=round(time.time() - started, 2),
            model=model,
        )

    def _build_payload(self, inputs: dict[str, Any]) -> tuple[dict[str, Any], str, str, str]:
        text = str(inputs.get("text", "")).strip()
        if not text:
            raise ValueError("MiMo TTS text must not be empty")
        model = str(inputs.get("model") or inputs.get("model_id") or self.MODEL)
        if model != self.MODEL:
            raise ValueError(f"MiMo TTS model must be {self.MODEL}")
        voice = str(inputs.get("voice") or inputs.get("voice_id") or "白桦")
        if voice not in self.VOICES:
            raise ValueError("MiMo TTS voice must be one of 冰糖, 茉莉, 苏打, 白桦, mimo_default")
        audio_format = str(inputs.get("format") or inputs.get("output_format") or "wav").lower()
        if audio_format != "wav":
            raise ValueError("MiMo TTS adapter currently supports WAV output only")

        messages: list[dict[str, str]] = []
        instructions = str(inputs.get("instructions", "")).strip()
        if instructions:
            messages.append({"role": "user", "content": instructions})
        # MiMo's speech API requires the text to synthesize in an assistant message.
        messages.append({"role": "assistant", "content": text})
        return (
            {
                "model": model,
                "messages": messages,
                "audio": {"format": audio_format, "voice": voice},
                "stream": False,
            },
            model,
            voice,
            audio_format,
        )

    @staticmethod
    def _extract_audio_data(response_data: dict[str, Any]) -> str:
        try:
            value = response_data["choices"][0]["message"]["audio"]["data"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("MiMo TTS response did not include choices[0].message.audio.data") from exc
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("MiMo TTS returned empty audio data")
        return value

    @staticmethod
    def _decode_audio(value: str) -> bytes:
        encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        encoded = "".join(encoded.split())
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise RuntimeError("MiMo TTS returned invalid base64 audio data") from exc

    @staticmethod
    def _validate_wav(content: bytes) -> None:
        if len(content) < 12 or not content.startswith(b"RIFF") or content[8:12] != b"WAVE":
            raise RuntimeError("MiMo TTS returned invalid WAV bytes")

    @staticmethod
    def _probe_duration(path: Path) -> float | None:
        try:
            from tools.analysis.audio_probe import probe_duration

            return probe_duration(path)
        except Exception:
            return None

    @staticmethod
    def _http_error(response: Any) -> str:
        message = ""
        try:
            data = response.json()
            error = data.get("error") if isinstance(data, dict) else None
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("code") or "")
            elif error:
                message = str(error)
            elif isinstance(data, dict):
                message = str(data.get("message") or data.get("code") or "")
        except Exception:
            message = ""
        suffix = f": {message[:300]}" if message else ""
        return f"MiMo API HTTP {response.status_code}{suffix}"

    @staticmethod
    def _safe_error(exc: Exception, api_key: str) -> str:
        message = str(exc).replace(api_key, "[redacted]") if api_key else str(exc)
        message = re.sub(r"(?i)(bearer|api-key[=: ]+)[^\s,;]+", r"\1 [redacted]", message)
        return message[:500]

