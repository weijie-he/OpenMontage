"""MiniMax Hailuo video generation.

The preferred backend calls MiniMax's official API directly with
``MINIMAX_API_KEY``.  The historical fal.ai gateway remains available as a
backward-compatible fallback for existing OpenMontage installations.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ResumeSupport,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class MiniMaxVideo(BaseTool):
    name = "minimax_video"
    version = "0.2.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "minimax"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set MINIMAX_API_KEY for MiniMax's official API (recommended).\n"
        "  Get one at https://platform.minimax.io/user-center/basic-information/interface-key\n"
        "Alternatively set FAL_KEY to use the legacy fal.ai gateway."
    )
    agent_skills = ["minimax", "ai-video-gen"]

    capabilities = ["text_to_video", "image_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "camera_direction": True,
        "official_api": True,
        "fal_gateway": True,
        "local_reference_image": True,
        "aspect_ratio_via_reference_image": True,
        "published_paygo_cost_estimate": True,
    }
    best_for = [
        "direct Hailuo 2.3 text-to-video and image-to-video generation",
        "prompt-following with explicit camera directions",
        "short cinematic clips with detailed character motion",
    ]
    not_good_for = [
        "offline generation",
        "single clips longer than 10 seconds",
        "direct text-to-video jobs that require an explicit aspect-ratio parameter",
    ]
    fallback_tools = ["kling_video", "veo_video", "wan_video"]

    input_schema = {
        "type": "object",
        "anyOf": [
            {"required": ["prompt"]},
            {"required": ["task_id"]},
            {"required": ["file_id"]},
            {"required": ["fal_status_url", "fal_response_url"]},
        ],
        "properties": {
            "prompt": {
                "type": "string",
                "maxLength": 2000,
                "description": "Scene and motion prompt. MiniMax camera commands such as [Static shot] are supported.",
            },
            "operation": {
                "type": "string",
                "enum": ["text_to_video", "image_to_video"],
                "default": "text_to_video",
            },
            "backend": {
                "type": "string",
                "enum": ["auto", "direct", "minimax", "fal"],
                "default": "auto",
                "description": "auto prefers MINIMAX_API_KEY and falls back to FAL_KEY; minimax aliases direct.",
            },
            "model": {
                "type": "string",
                "enum": [
                    "MiniMax-Hailuo-2.3",
                    "MiniMax-Hailuo-2.3-Fast",
                    "MiniMax-Hailuo-02",
                ],
                "default": "MiniMax-Hailuo-2.3",
                "description": "Official MiniMax model used by the direct backend.",
            },
            "model_variant": {
                "type": "string",
                "enum": [
                    "hailuo-02/pro",
                    "hailuo-02/standard",
                    "hailuo-2.3/pro",
                    "hailuo-2.3/standard",
                    "hailuo-2.3-fast/pro",
                    "hailuo-2.3-fast/standard",
                ],
                "default": "hailuo-02/pro",
                "description": "fal.ai model variant; retained for backward compatibility.",
            },
            "duration": {
                "type": ["integer", "string"],
                "enum": [6, 10, "6", "10"],
                "default": 6,
            },
            "resolution": {
                "type": "string",
                "enum": ["512P", "768P", "1080P"],
                "default": "1080P",
            },
            "prompt_optimizer": {"type": "boolean", "default": True},
            "fast_pretreatment": {"type": "boolean", "default": False},
            "first_frame_image": {
                "type": "string",
                "description": "Public image URL or Base64 data URL for image-to-video.",
            },
            "image_url": {
                "type": "string",
                "description": "Alias for first_frame_image/reference_image_url.",
            },
            "reference_image_url": {"type": "string"},
            "reference_image_path": {
                "type": "string",
                "description": "Local first-frame image. Direct mode encodes it as a Base64 data URL.",
            },
            "poll_interval_seconds": {
                "type": "number",
                "minimum": 1,
                "default": 10,
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 30,
                "default": 1800,
            },
            "task_id": {
                "type": "string",
                "description": "Resume an existing official MiniMax task without submitting a new paid task.",
            },
            "file_id": {
                "type": "string",
                "description": "Resume official MiniMax file retrieval without submitting or polling a task.",
            },
            "fal_status_url": {
                "type": "string",
                "description": "Resume polling a previously submitted fal.ai task.",
            },
            "fal_response_url": {
                "type": "string",
                "description": "Result URL paired with fal_status_url for fal.ai resume.",
            },
            "output_path": {"type": "string"},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "output": {"type": "string"},
            "backend": {"type": "string"},
            "task_id": {"type": ["string", "null"]},
            "file_id": {"type": ["string", "null"]},
            "video_duration_seconds": {"type": ["number", "null"]},
            "submission_attempted": {"type": "boolean"},
            "submitted_this_call": {"type": "boolean"},
            "charge_state": {"type": "string"},
            "resubmit_safe": {"type": "boolean"},
            "potential_cost_usd": {"type": "number"},
            "resume_inputs": {"type": "object"},
        },
    }
    artifact_schema = {"type": "array", "items": {"type": "string"}}

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(
        # Video submission is paid and asynchronous. Blind execution retries
        # can create duplicate charged tasks; resume by task_id instead.
        max_retries=0,
        backoff_seconds=0.0,
        retryable_errors=[],
    )
    resume_support = ResumeSupport.FROM_CHECKPOINT
    idempotency_key_fields = [
        "prompt",
        "backend",
        "model",
        "model_variant",
        "operation",
        "duration",
        "resolution",
        "reference_image_path",
        "reference_image_url",
        "task_id",
        "file_id",
    ]
    side_effects = [
        "writes video file to output_path",
        "calls the MiniMax official API or fal.ai gateway",
    ]
    user_visible_verification = [
        "Watch the generated clip for motion coherence, character consistency, and prompt adherence"
    ]

    DIRECT_CREATE_URL = "https://api.minimax.io/v1/video_generation"
    DIRECT_QUERY_URL = "https://api.minimax.io/v1/query/video_generation"
    DIRECT_RETRIEVE_URL = "https://api.minimax.io/v1/files/retrieve"
    FAL_QUEUE_HOSTS = {"queue.fal.run"}
    MAX_REFERENCE_IMAGE_BYTES = 20 * 1024 * 1024
    DIRECT_T2V_MODELS = {
        "MiniMax-Hailuo-2.3",
        "MiniMax-Hailuo-02",
    }
    DIRECT_I2V_MODELS = {
        "MiniMax-Hailuo-2.3",
        "MiniMax-Hailuo-2.3-Fast",
        "MiniMax-Hailuo-02",
    }
    # Official pay-as-you-go prices verified against MiniMax's pricing page.
    # Environment overrides below remain useful for negotiated/package rates.
    DIRECT_PAYGO_PRICES_USD = {
        ("MiniMax-Hailuo-2.3-Fast", "768P", 6): 0.19,
        ("MiniMax-Hailuo-2.3-Fast", "768P", 10): 0.32,
        ("MiniMax-Hailuo-2.3-Fast", "1080P", 6): 0.33,
        ("MiniMax-Hailuo-2.3", "768P", 6): 0.28,
        ("MiniMax-Hailuo-2.3", "768P", 10): 0.56,
        ("MiniMax-Hailuo-2.3", "1080P", 6): 0.49,
        ("MiniMax-Hailuo-02", "512P", 6): 0.10,
        ("MiniMax-Hailuo-02", "512P", 10): 0.15,
        ("MiniMax-Hailuo-02", "768P", 6): 0.28,
        ("MiniMax-Hailuo-02", "768P", 10): 0.56,
        ("MiniMax-Hailuo-02", "1080P", 6): 0.49,
    }
    FAL_PAYGO_PRICES_USD = {
        ("hailuo-2.3/pro", 6): 0.49,
        ("hailuo-2.3-fast/pro", 6): 0.33,
        ("hailuo-2.3/standard", 6): 0.28,
        ("hailuo-2.3/standard", 10): 0.56,
        ("hailuo-2.3-fast/standard", 6): 0.19,
        ("hailuo-2.3-fast/standard", 10): 0.32,
    }
    FAL_SUPPORTED_VARIANTS = {
        "hailuo-02/pro",
        "hailuo-02/standard",
        "hailuo-2.3/pro",
        "hailuo-2.3/standard",
        "hailuo-2.3-fast/pro",
        "hailuo-2.3-fast/standard",
    }

    @staticmethod
    def _direct_api_key() -> str | None:
        return os.environ.get("MINIMAX_API_KEY")

    @staticmethod
    def _fal_api_key() -> str | None:
        return os.environ.get("FAL_KEY") or os.environ.get("FAL_AI_API_KEY")

    def _resolve_backend(self, inputs: dict[str, Any]) -> str:
        raw_backend = inputs["backend"] if "backend" in inputs else "auto"
        if not isinstance(raw_backend, str):
            raise ValueError(
                "MiniMax backend must be one of auto, direct, minimax, or fal"
            )
        requested = raw_backend.lower()
        if requested == "minimax":
            return "direct"
        if requested in {"direct", "fal"}:
            return requested
        if requested != "auto":
            raise ValueError(
                f"MiniMax backend={raw_backend!r} is invalid; use auto, direct, minimax, or fal"
            )
        # Resume identity is stronger than newly available credentials. This
        # prevents backend="auto" from switching a prior fal task to direct
        # (or vice versa) and submitting a second paid task.
        has_direct_resume = bool(inputs.get("task_id") or inputs.get("file_id"))
        has_fal_resume = bool(inputs.get("fal_status_url") or inputs.get("fal_response_url"))
        if has_direct_resume and not has_fal_resume:
            return "direct"
        if has_fal_resume and not has_direct_resume:
            return "fal"
        if self._direct_api_key():
            return "direct"
        if self._fal_api_key():
            return "fal"
        # Report the preferred setup path when no credentials are configured.
        return "direct"

    def get_status(self) -> ToolStatus:
        if self._direct_api_key() or self._fal_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        backend = self._resolve_backend(inputs)
        resume_error = self._resume_backend_error(inputs, backend)
        if resume_error:
            raise ValueError(resume_error)
        if backend == "direct" and (inputs.get("task_id") or inputs.get("file_id")):
            return 0.0
        if backend == "fal" and inputs.get("fal_status_url") and inputs.get("fal_response_url"):
            return 0.0
        if backend == "fal":
            variant = str(inputs.get("model_variant", "hailuo-02/pro"))
            duration = self._duration(inputs)
            known = self.FAL_PAYGO_PRICES_USD.get((variant, duration))
            if known is not None:
                return known
            if variant == "hailuo-02/pro":
                return round(0.08 * duration, 4)
            if variant == "hailuo-02/standard":
                resolution = str(inputs.get("resolution", "768P")).upper()
                is_512_i2v = (
                    str(inputs.get("operation", "text_to_video")) == "image_to_video"
                    and resolution == "512P"
                )
                return round((0.017 if is_512_i2v else 0.045) * duration, 4)
            # Preserve the historical Video-01 estimate when that legacy model
            # is selected; its current price is not exposed in the Hailuo pages.
            return 0.15

        operation = str(inputs.get("operation", "text_to_video"))
        model = str(inputs.get("model", "MiniMax-Hailuo-2.3"))
        resolution = str(inputs.get("resolution", "1080P")).upper()
        duration = self._duration(inputs)
        self._validate_direct_spec(
            operation=operation,
            model=model,
            duration=duration,
            resolution=resolution,
        )
        # Overrides take precedence for negotiated rates or prepaid packages.
        per_clip = self._float_env("MINIMAX_VIDEO_COST_PER_CLIP_USD")
        if per_clip is not None:
            return round(per_clip, 4)
        per_second = self._float_env("MINIMAX_VIDEO_COST_PER_SECOND_USD")
        if per_second is not None:
            return round(per_second * duration, 4)
        return self.DIRECT_PAYGO_PRICES_USD.get(
            (model, resolution, duration),
            0.0,
        )

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        if self._resolve_backend(inputs) == "fal":
            variant = str(inputs.get("model_variant", "hailuo-02/pro"))
            return 30.0 if "fast" in variant else 60.0
        return 240.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        try:
            backend = self._resolve_backend(inputs)
        except ValueError as exc:
            return ToolResult(
                success=False,
                data={
                    "provider": self.provider,
                    "backend": inputs.get("backend"),
                    "submission_attempted": False,
                    "submitted_this_call": False,
                    "charge_state": "not_submitted",
                    "resubmit_safe": True,
                    "cost_estimate_status": "invalid_backend",
                },
                error=f"MiniMax video request rejected: {self._safe_error(exc)}",
            )
        resume_error = self._resume_backend_error(inputs, backend)
        if resume_error:
            return ToolResult(
                success=False,
                data={
                    "provider": self.provider,
                    "backend": backend,
                    "submitted_this_call": False,
                    "cost_estimate_status": "invalid_cross_backend_resume",
                },
                error=f"MiniMax video resume rejected: {resume_error}",
            )
        if backend == "direct":
            api_key = self._direct_api_key()
            if not api_key:
                return ToolResult(
                    success=False,
                    error="MINIMAX_API_KEY not set. " + self.install_instructions,
                )
            return self._execute_direct(inputs, api_key)

        api_key = self._fal_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="FAL_KEY not set for backend='fal'. " + self.install_instructions,
            )
        return self._execute_fal(inputs, api_key)

    def _execute_direct(self, inputs: dict[str, Any], api_key: str) -> ToolResult:
        import requests

        started = time.time()
        task_id = str(inputs.get("task_id") or "") or None
        file_id = str(inputs.get("file_id") or "") or None
        resuming_this_call = task_id is not None or file_id is not None
        submission_attempted = False
        submitted_this_call = False
        query_data: dict[str, Any] = {}
        payload: dict[str, Any] | None = None
        output_path = Path(inputs.get("output_path", "minimax_output.mp4"))
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            poll_interval = float(inputs.get("poll_interval_seconds", 10))
            timeout_seconds = int(inputs.get("timeout_seconds", 1800))
            if file_id is None:
                if poll_interval < 1:
                    raise ValueError("MiniMax poll_interval_seconds must be at least 1")
                if timeout_seconds < 30:
                    raise ValueError("MiniMax timeout_seconds must be at least 30")
            if task_id is None and file_id is None:
                if not inputs.get("prompt"):
                    raise ValueError("MiniMax video requires prompt, task_id, or file_id")
                payload = self._build_direct_payload(inputs)
                submission_attempted = True
                submit_response = requests.post(
                    self.DIRECT_CREATE_URL,
                    headers=headers,
                    json=payload,
                    timeout=(10, 60),
                )
                submit_response.raise_for_status()
                submit_data = submit_response.json()
                self._raise_for_api_error(submit_data, "create task")
                task_id = (
                    submit_data.get("task_id")
                    or submit_data.get("video_generation_resp", {}).get("task_id")
                    or submit_data.get("data", {}).get("task_id")
                )
                if not task_id:
                    raise RuntimeError("MiniMax create task response did not include task_id")
                task_id = str(task_id)
                submitted_this_call = True

            if file_id is None:
                if task_id is None:
                    raise RuntimeError("MiniMax resume requires task_id or file_id")
                query_data = self._poll_direct_task(
                    requests_module=requests,
                    headers=headers,
                    task_id=task_id,
                    poll_interval=poll_interval,
                    timeout_seconds=timeout_seconds,
                )
                file_id = query_data.get("file_id")
                if not file_id:
                    raise RuntimeError("MiniMax task succeeded but did not include file_id")
                file_id = str(file_id)

            retrieve_response = requests.get(
                self.DIRECT_RETRIEVE_URL,
                headers=headers,
                params={"file_id": file_id},
                timeout=(10, 60),
            )
            retrieve_response.raise_for_status()
            retrieve_data = retrieve_response.json()
            self._raise_for_api_error(retrieve_data, "retrieve file")
            download_url = retrieve_data.get("file", {}).get("download_url")
            if not download_url:
                raise RuntimeError("MiniMax file response did not include file.download_url")

            download = requests.get(download_url, timeout=(10, 180))
            download.raise_for_status()
            self._validate_video_bytes(download.content)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(download.content)
        except Exception as exc:
            estimated_cost = (
                0.0
                if resuming_this_call
                else self._safe_cost_estimate({**inputs, "backend": "direct"})
            )
            error_data: dict[str, Any] = {
                "provider": self.provider,
                "backend": "direct",
                "task_id": task_id,
                "file_id": file_id,
                "submission_attempted": submission_attempted,
                "submitted_this_call": submitted_this_call,
                "estimated_cost_usd": estimated_cost,
                "potential_cost_usd": estimated_cost if submission_attempted else 0.0,
                "charge_state": (
                    "resume_no_new_charge"
                    if resuming_this_call
                    else (
                        "submitted_estimated"
                        if submitted_this_call
                        else ("unknown" if submission_attempted else "not_submitted")
                    )
                ),
                "resubmit_safe": resuming_this_call or not submission_attempted,
                "cost_estimate_status": (
                    "resume_no_new_charge"
                    if resuming_this_call
                    else self._direct_cost_estimate_status(inputs)
                ),
            }
            if task_id or file_id:
                resume_inputs: dict[str, Any] = {
                    "backend": "direct",
                    "output_path": str(output_path),
                }
                if task_id:
                    resume_inputs["task_id"] = task_id
                if file_id:
                    resume_inputs["file_id"] = file_id
                for key in ("operation", "model", "duration", "resolution"):
                    if payload and key in payload:
                        resume_inputs[key] = payload[key]
                    elif key in inputs:
                        resume_inputs[key] = inputs[key]
                error_data["resume_inputs"] = resume_inputs
            retry_warning = (
                " Submission was attempted but no resumable task ID was obtained; "
                "charge state is unknown and the prompt must not be resubmitted blindly."
                if submission_attempted and not task_id and not file_id
                else ""
            )
            return ToolResult(
                success=False,
                data=error_data,
                error=(
                    f"MiniMax direct video generation failed: {self._safe_error(exc)}"
                    f"{retry_warning}"
                ),
                cost_usd=estimated_cost if submitted_this_call else 0.0,
                duration_seconds=round(time.time() - started, 2),
                model=str(inputs["model"]) if inputs.get("model") else None,
            )

        cost = self.estimate_cost({**inputs, "backend": "direct"})
        model = payload["model"] if payload else inputs.get("model")
        duration = (
            payload["duration"]
            if payload
            else (self._duration(inputs) if "duration" in inputs else None)
        )
        resolution = (
            payload["resolution"]
            if payload
            else (str(inputs["resolution"]).upper() if "resolution" in inputs else None)
        )
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "backend": "direct",
                "model": model,
                "prompt": inputs.get("prompt"),
                "task_id": task_id,
                "file_id": file_id,
                "submission_attempted": submission_attempted,
                "submitted_this_call": submitted_this_call,
                "charge_state": (
                    "submitted_estimated" if submitted_this_call else "resume_no_new_charge"
                ),
                "resubmit_safe": False,
                "potential_cost_usd": cost if submitted_this_call else 0.0,
                "video_width": query_data.get("video_width"),
                "video_height": query_data.get("video_height"),
                "video_duration_seconds": duration,
                "resolution": resolution,
                "output": str(output_path),
                "cost_estimate_status": (
                    self._direct_cost_estimate_status(inputs)
                    if submitted_this_call
                    else "resume_no_new_charge"
                ),
            },
            artifacts=[str(output_path)],
            cost_usd=cost if submitted_this_call else 0.0,
            duration_seconds=round(time.time() - started, 2),
            model=str(model) if model else None,
        )

    def _build_direct_payload(self, inputs: dict[str, Any]) -> dict[str, Any]:
        operation = str(inputs.get("operation", "text_to_video"))
        model = str(inputs.get("model", "MiniMax-Hailuo-2.3"))
        duration = self._duration(inputs)
        resolution = str(inputs.get("resolution", "1080P")).upper()

        if len(str(inputs["prompt"])) > 2000:
            raise ValueError("MiniMax prompt must be 2000 characters or fewer")
        self._validate_direct_spec(
            operation=operation,
            model=model,
            duration=duration,
            resolution=resolution,
        )

        payload: dict[str, Any] = {
            "model": model,
            "prompt": inputs["prompt"],
            "duration": duration,
            "resolution": resolution,
            "prompt_optimizer": bool(inputs.get("prompt_optimizer", True)),
        }
        if inputs.get("fast_pretreatment"):
            if model not in {
                "MiniMax-Hailuo-2.3",
                "MiniMax-Hailuo-2.3-Fast",
                "MiniMax-Hailuo-02",
            }:
                raise ValueError(f"fast_pretreatment is not supported by {model}")
            payload["fast_pretreatment"] = True

        if operation == "image_to_video":
            first_frame = self._first_frame_image(inputs)
            if not first_frame:
                raise ValueError(
                    "image_to_video requires first_frame_image, image_url, "
                    "reference_image_url, or reference_image_path"
                )
            payload["first_frame_image"] = first_frame
        return payload

    def _poll_direct_task(
        self,
        *,
        requests_module: Any,
        headers: dict[str, str],
        task_id: str,
        poll_interval: float,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        if poll_interval < 1:
            raise ValueError("MiniMax poll_interval_seconds must be at least 1")
        if timeout_seconds < 30:
            raise ValueError("MiniMax timeout_seconds must be at least 30")
        deadline = time.monotonic() + timeout_seconds
        while True:
            response = requests_module.get(
                self.DIRECT_QUERY_URL,
                headers=headers,
                params={"task_id": task_id},
                timeout=(10, 30),
            )
            response.raise_for_status()
            data = response.json()
            self._raise_for_api_error(data, "query task")
            status = str(data.get("status", "")).lower()
            if status == "success":
                return data
            if status == "fail":
                message = data.get("base_resp", {}).get("status_msg") or "task failed"
                raise RuntimeError(f"MiniMax video task failed: {message}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"MiniMax video task {task_id} did not finish within {timeout_seconds}s"
                )
            time.sleep(max(0.0, poll_interval))

    def _execute_fal(self, inputs: dict[str, Any], api_key: str) -> ToolResult:
        import requests

        started = time.time()
        operation = str(inputs.get("operation", "text_to_video"))
        variant = str(inputs.get("model_variant", "hailuo-02/pro"))
        model_path = self._fal_model_path(operation, variant)
        status_url = str(inputs.get("fal_status_url") or "") or None
        response_url = str(inputs.get("fal_response_url") or "") or None
        resuming_this_call = bool(status_url and response_url)
        request_id: str | None = None
        submission_attempted = False
        submitted_this_call = False
        payload: dict[str, Any] = {}
        output_path = Path(inputs.get("output_path", "minimax_output.mp4"))

        try:
            if bool(status_url) != bool(response_url):
                raise ValueError(
                    "fal.ai resume requires both fal_status_url and fal_response_url; refusing to resubmit"
                )
            if status_url and response_url:
                status_url = self._validate_fal_queue_url(status_url, "fal_status_url")
                response_url = self._validate_fal_queue_url(response_url, "fal_response_url")
            timeout_seconds = int(inputs.get("timeout_seconds", 1800))
            poll_interval = float(inputs.get("poll_interval_seconds", 5))
            if timeout_seconds < 30:
                raise ValueError("MiniMax timeout_seconds must be at least 30")
            if poll_interval < 1:
                raise ValueError("MiniMax poll_interval_seconds must be at least 1")
            headers = {
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json",
            }
            if not status_url or not response_url:
                if not inputs.get("prompt"):
                    raise ValueError(
                        "fal.ai MiniMax requires prompt or both fal_status_url and fal_response_url"
                    )
                payload = self._build_fal_payload(inputs, operation=operation, variant=variant)
                submission_attempted = True
                submit_response = requests.post(
                    f"https://queue.fal.run/fal-ai/{model_path}",
                    headers=headers,
                    json=payload,
                    timeout=(10, 30),
                )
                submit_response.raise_for_status()
                queue_data = submit_response.json()
                request_id = queue_data.get("request_id")
                # The task exists after a successful queue response even if a
                # malformed provider URL prevents polling it locally.
                submitted_this_call = True
                status_url = self._validate_fal_queue_url(
                    str(queue_data["status_url"]), "submitted status_url"
                )
                response_url = self._validate_fal_queue_url(
                    str(queue_data["response_url"]), "submitted response_url"
                )

            deadline = time.monotonic() + timeout_seconds
            while True:
                status_response = requests.get(status_url, headers=headers, timeout=(10, 30))
                status_response.raise_for_status()
                status = status_response.json().get("status", "UNKNOWN")
                if status == "COMPLETED":
                    break
                if status in {"FAILED", "CANCELLED"}:
                    raise RuntimeError(f"fal.ai MiniMax task {status.lower()}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("fal.ai MiniMax video task timed out")
                time.sleep(poll_interval)

            result_response = requests.get(response_url, headers=headers, timeout=(10, 60))
            result_response.raise_for_status()
            data = result_response.json()
            video_url = data.get("video", {}).get("url") or data.get("video_url")
            if not video_url:
                raise RuntimeError("fal.ai MiniMax response did not include a video URL")

            download = requests.get(video_url, timeout=(10, 180))
            download.raise_for_status()
            self._validate_video_bytes(download.content)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(download.content)
        except Exception as exc:
            estimated_cost = self._safe_cost_estimate({**inputs, "backend": "fal"})
            error_data: dict[str, Any] = {
                "provider": self.provider,
                "backend": "fal",
                "request_id": request_id,
                "fal_status_url": status_url,
                "fal_response_url": response_url,
                "submission_attempted": submission_attempted,
                "submitted_this_call": submitted_this_call,
                "estimated_cost_usd": estimated_cost,
                "potential_cost_usd": estimated_cost if submission_attempted else 0.0,
                "charge_state": (
                    "resume_no_new_charge"
                    if resuming_this_call
                    else (
                        "submitted_estimated"
                        if submitted_this_call
                        else ("unknown" if submission_attempted else "not_submitted")
                    )
                ),
                "resubmit_safe": resuming_this_call or not submission_attempted,
                "cost_estimate_status": (
                    "resume_no_new_charge"
                    if resuming_this_call
                    else "published_paygo"
                ),
            }
            if status_url and response_url:
                error_data["resume_inputs"] = {
                    "backend": "fal",
                    "fal_status_url": status_url,
                    "fal_response_url": response_url,
                    "model_variant": variant,
                    "operation": operation,
                    "output_path": str(output_path),
                }
            retry_warning = (
                " Submission was attempted but no resumable fal.ai queue URLs were obtained; "
                "charge state is unknown and the prompt must not be resubmitted blindly."
                if submission_attempted and not status_url and not response_url
                else ""
            )
            return ToolResult(
                success=False,
                data=error_data,
                error=(
                    f"MiniMax fal.ai video generation failed: {self._safe_error(exc)}"
                    f"{retry_warning}"
                ),
                cost_usd=estimated_cost if submitted_this_call else 0.0,
                duration_seconds=round(time.time() - started, 2),
                model=f"fal-ai/{model_path}",
            )

        generated_duration = data.get("video", {}).get("duration")
        if generated_duration is None and "duration" in payload:
            generated_duration = int(payload["duration"])
        cost = self.estimate_cost({**inputs, "backend": "fal"})
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "backend": "fal",
                "model": f"fal-ai/{model_path}",
                "prompt": inputs.get("prompt"),
                "request_id": request_id,
                "submission_attempted": submission_attempted,
                "submitted_this_call": submitted_this_call,
                "charge_state": (
                    "submitted_estimated" if submitted_this_call else "resume_no_new_charge"
                ),
                "resubmit_safe": False,
                "potential_cost_usd": cost if submitted_this_call else 0.0,
                "video_duration_seconds": generated_duration,
                "output": str(output_path),
                "cost_estimate_status": (
                    "published_paygo" if submitted_this_call else "resume_no_new_charge"
                ),
            },
            artifacts=[str(output_path)],
            cost_usd=cost if submitted_this_call else 0.0,
            duration_seconds=round(time.time() - started, 2),
            model=f"fal-ai/{model_path}",
        )

    def _build_fal_payload(
        self,
        inputs: dict[str, Any],
        *,
        operation: str,
        variant: str,
    ) -> dict[str, Any]:
        duration = self._duration(inputs)
        if variant not in self.FAL_SUPPORTED_VARIANTS:
            raise ValueError(
                f"fal.ai MiniMax variant {variant} has no configured cost contract"
            )
        if operation == "text_to_video" and variant.startswith("hailuo-2.3-fast/"):
            raise ValueError("fal.ai Hailuo 2.3 Fast supports image_to_video only")

        payload: dict[str, Any] = {
            "prompt": inputs["prompt"],
            "prompt_optimizer": bool(inputs.get("prompt_optimizer", True)),
        }
        if operation == "image_to_video":
            image_url = (
                inputs.get("image_url")
                or inputs.get("reference_image_url")
                or inputs.get("first_frame_image")
            )
            if not image_url and inputs.get("reference_image_path"):
                from tools.video._shared import upload_image_fal

                image_url = upload_image_fal(str(inputs["reference_image_path"]))
            if not image_url:
                raise ValueError("fal image_to_video requires an image URL or local reference image")
            payload["image_url"] = image_url

        if variant in {
            "hailuo-02/pro",
            "hailuo-2.3/pro",
            "hailuo-2.3-fast/pro",
        }:
            if "duration" in inputs and duration != 6:
                raise ValueError(
                    f"fal.ai {variant} is a fixed Pro endpoint and does not accept duration=10"
                )
            if "resolution" in inputs and str(inputs["resolution"]).upper() != "1080P":
                raise ValueError(f"fal.ai {variant} is the fixed 1080P Pro endpoint")
            return payload

        payload["duration"] = str(duration)
        if "resolution" in inputs:
            resolution = str(inputs["resolution"]).upper()
            expected = "1080P" if variant.endswith("/pro") else "768P"
            if variant == "hailuo-02/standard" and operation == "image_to_video":
                if resolution not in {"512P", "768P"}:
                    raise ValueError(
                        "fal.ai hailuo-02/standard image_to_video expects 512P or 768P"
                    )
                payload["resolution"] = resolution
            elif variant == "hailuo-02/standard" and operation == "text_to_video":
                if resolution != "768P":
                    raise ValueError(
                        "fal.ai hailuo-02/standard text_to_video is fixed at 768P"
                    )
            elif resolution != expected:
                raise ValueError(f"fal.ai {variant} expects {expected}")
        return payload

    @staticmethod
    def _fal_model_path(operation: str, variant: str) -> str:
        if operation == "text_to_video":
            return "minimax/video-01" if variant == "video-01" else f"minimax/{variant}/text-to-video"
        return (
            "minimax/video-01/image-to-video"
            if variant == "video-01"
            else f"minimax/{variant}/image-to-video"
        )

    def _first_frame_image(self, inputs: dict[str, Any]) -> str | None:
        remote = (
            inputs.get("first_frame_image")
            or inputs.get("image_url")
            or inputs.get("reference_image_url")
        )
        if remote:
            return str(remote)

        local = inputs.get("reference_image_path")
        if not local:
            return None
        path = Path(str(local)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MiniMax reference image not found: {path}")
        if path.stat().st_size >= self.MAX_REFERENCE_IMAGE_BYTES:
            raise ValueError("MiniMax reference image must be smaller than 20 MB")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("MiniMax reference image must be JPG, JPEG, PNG, or WebP")
        try:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
        except Exception as exc:
            raise ValueError(f"MiniMax reference image is invalid: {path}") from exc
        if min(width, height) <= 300:
            raise ValueError("MiniMax reference image short edge must be greater than 300px")
        aspect_ratio = width / height
        if not 0.4 <= aspect_ratio <= 2.5:
            raise ValueError("MiniMax reference image aspect ratio must be between 2:5 and 5:2")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _validate_direct_spec(
        self,
        *,
        operation: str,
        model: str,
        duration: int,
        resolution: str,
    ) -> None:
        if operation not in {"text_to_video", "image_to_video"}:
            raise ValueError("MiniMax operation must be text_to_video or image_to_video")
        supported_models = (
            self.DIRECT_T2V_MODELS
            if operation == "text_to_video"
            else self.DIRECT_I2V_MODELS
        )
        if model not in supported_models:
            raise ValueError(f"MiniMax model {model} does not support {operation}")
        if duration not in {6, 10}:
            raise ValueError("MiniMax duration must be 6 or 10 seconds")

        if model == "MiniMax-Hailuo-02" and operation == "image_to_video":
            allowed_resolutions = {"512P", "768P", "1080P"}
        elif model in {
            "MiniMax-Hailuo-2.3",
            "MiniMax-Hailuo-2.3-Fast",
            "MiniMax-Hailuo-02",
        }:
            allowed_resolutions = {"768P", "1080P"}
        else:
            allowed_resolutions = {"720P"}

        if resolution not in allowed_resolutions:
            allowed = ", ".join(sorted(allowed_resolutions))
            raise ValueError(f"MiniMax {model} supports resolution(s): {allowed}")
        if duration == 10 and resolution == "1080P":
            raise ValueError("MiniMax 1080P generation supports 6-second clips only")
        if duration == 10 and model not in {
            "MiniMax-Hailuo-2.3",
            "MiniMax-Hailuo-2.3-Fast",
            "MiniMax-Hailuo-02",
        }:
            raise ValueError(f"MiniMax {model} supports 6-second clips only")

    def _direct_cost_estimate_status(self, inputs: dict[str, Any]) -> str:
        if self._float_env("MINIMAX_VIDEO_COST_PER_CLIP_USD") is not None:
            return "configured_override"
        if self._float_env("MINIMAX_VIDEO_COST_PER_SECOND_USD") is not None:
            return "configured_override"
        try:
            key = (
                str(inputs.get("model", "MiniMax-Hailuo-2.3")),
                str(inputs.get("resolution", "1080P")).upper(),
                self._duration(inputs),
            )
        except ValueError:
            return "unconfigured"
        return "published_paygo" if key in self.DIRECT_PAYGO_PRICES_USD else "unconfigured"

    def _safe_cost_estimate(self, inputs: dict[str, Any]) -> float:
        try:
            return self.estimate_cost(inputs)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _raise_for_api_error(data: dict[str, Any], context: str) -> None:
        base_resp = data.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0, "0"):
            status_msg = base_resp.get("status_msg") or "unknown API error"
            raise RuntimeError(f"MiniMax {context} error {status_code}: {status_msg}")

    @staticmethod
    def _validate_video_bytes(content: bytes) -> None:
        if not content:
            raise RuntimeError("MiniMax video download returned an empty body")
        if b"ftyp" not in content[:64]:
            raise RuntimeError(
                "MiniMax video download was not an MP4 file (missing ftyp header)"
            )

    @staticmethod
    def _duration(inputs: dict[str, Any]) -> int:
        try:
            return int(inputs.get("duration", 6))
        except (TypeError, ValueError) as exc:
            raise ValueError("MiniMax duration must be 6 or 10 seconds") from exc

    @staticmethod
    def _float_env(name: str) -> float | None:
        raw = os.environ.get(name)
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    @staticmethod
    def _resume_backend_error(inputs: dict[str, Any], backend: str) -> str | None:
        has_direct_resume = bool(inputs.get("task_id") or inputs.get("file_id"))
        has_fal_resume = bool(inputs.get("fal_status_url") or inputs.get("fal_response_url"))
        if has_direct_resume and has_fal_resume:
            return "official MiniMax task/file IDs cannot be mixed with fal.ai resume URLs"
        if backend == "direct" and has_fal_resume:
            return "backend='direct' cannot consume fal.ai resume URLs"
        if backend == "fal" and has_direct_resume:
            return "backend='fal' cannot consume official MiniMax task_id/file_id values"
        return None

    @classmethod
    def _validate_fal_queue_url(cls, value: str, field_name: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"{field_name} is not a valid fal.ai queue URL") from exc
        if (
            parsed.scheme != "https"
            or parsed.hostname not in cls.FAL_QUEUE_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or port not in (None, 443)
        ):
            raise ValueError(
                f"{field_name} must be an HTTPS URL on an approved fal.ai queue host"
            )
        return value

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        message = str(exc)
        for key_name in ("MINIMAX_API_KEY", "FAL_KEY", "FAL_AI_API_KEY"):
            key = os.environ.get(key_name)
            if key:
                message = message.replace(key, "[redacted]")
        # Signed CDN/queue URLs may surface through requests.HTTPError. Keep the
        # endpoint useful for diagnosis while removing every query value.
        message = re.sub(
            r"([?&][^=\s&]+)=([^&\s]+)",
            r"\1=[redacted]",
            message,
        )
        return message
