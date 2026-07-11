---
name: minimax
description: MiniMax official API integration for Hailuo 2.3 text/image-to-video and Speech 2.8 text-to-speech. Use when generating video or narration directly with MINIMAX_API_KEY, when keeping video and TTS under one MiniMax account, or when debugging MiniMax task polling and hexadecimal audio responses.
---

# MiniMax Video + TTS

Use the official MiniMax API with bearer authentication:

```text
Authorization: Bearer ${MINIMAX_API_KEY}
Base URL: https://api.minimax.io
```

Never log the key or include it in project artifacts. The OpenMontage tools redact it from returned errors.

## OpenMontage Tools

| Need | Tool | Default model |
|---|---|---|
| Text/image to video | `minimax_video` | `MiniMax-Hailuo-2.3` |
| Text to speech | `minimax_tts` | `speech-2.8-hd` |

`minimax_video` supports `backend="auto"`, `backend="direct"` (also accepted as `"minimax"`), and the legacy `backend="fal"`. Direct is the preferred route. In `auto`, `MINIMAX_API_KEY` takes priority over `FAL_KEY`.

When the user explicitly selects MiniMax, route with both:

```python
preferred_provider="minimax"
allowed_providers=["minimax"]
backend="direct"
```

This prevents a scoring difference from switching providers and prevents the legacy fal.ai gateway from being selected when official direct routing was approved.

## Video API

Video generation is asynchronous and has three steps:

```text
POST https://api.minimax.io/v1/video_generation
GET  https://api.minimax.io/v1/query/video_generation?task_id=...
GET  https://api.minimax.io/v1/files/retrieve?file_id=...
```

The final retrieve response contains `file.download_url`; download it immediately rather than treating the temporary URL as the artifact.

### Text to video

```json
{
  "model": "MiniMax-Hailuo-2.3",
  "prompt": "A girl looks across the river [Static shot].",
  "duration": 6,
  "resolution": "1080P",
  "prompt_optimizer": true
}
```

### Image to video

Add `first_frame_image`. The value may be a public URL or a Base64 data URL. OpenMontage accepts `reference_image_path` and creates the data URL locally; do not upload that image through fal.ai when the direct backend is selected.

Official limits used by the tool:

- Prompt: at most 2,000 characters.
- Duration: 6 or 10 seconds.
- 1080P: 6 seconds; use 768P for a 10-second clip.
- Local first-frame image: less than 20 MB.
- `MiniMax-Hailuo-2.3-Fast`: image-to-video only.

Task statuses are `Preparing`, `Queueing`, `Processing`, `Success`, and `Fail`. Poll about every 10 seconds and always use a finite timeout.

Never blindly resubmit after a polling or download timeout: the paid task may still be running. The tool returns `task_id`/`file_id` plus `resume_inputs`; call it again with those IDs to resume without creating a duplicate task.

If the initial POST times out before a task ID is received, the tool reports
`charge_state="unknown"`, `potential_cost_usd`, and `resubmit_safe=false`.
Treat that as a billing incident to reconcile in the MiniMax console; do not
repeat the prompt merely because no local artifact exists. Official task IDs
and fal.ai queue URLs are backend-specific and must never be mixed. fal.ai
resume URLs are accepted only on the official `queue.fal.run` HTTPS host so an
API key cannot be forwarded to an arbitrary URL.

Camera commands can be placed directly in the prompt, for example `[Static shot]`, `[Pan left]`, or `[Pedestal up]`. For character work, keep each clip to one principal action and describe motion rather than repeating the still image's appearance.

## Speech API

Use non-streaming HTTP synthesis:

```text
POST https://api.minimax.io/v1/t2a_v2
```

```json
{
  "model": "speech-2.8-hd",
  "text": "天保，你又来了。",
  "stream": false,
  "language_boost": "auto",
  "output_format": "hex",
  "voice_setting": {
    "voice_id": "male-qn-qingse",
    "speed": 1.0,
    "vol": 1.0,
    "pitch": 0
  },
  "audio_setting": {
    "sample_rate": 32000,
    "bitrate": 128000,
    "format": "mp3",
    "channel": 1
  }
}
```

The audio is returned as hexadecimal text in `data.audio`. Decode it with `bytes.fromhex()` before writing the file. HTTP 200 is not enough: also require `base_resp.status_code == 0`.

Supported OpenMontage controls include `voice_id`, `speed`/`speaking_rate`, `volume`, `pitch`, `emotion`, `language_boost`, `pronunciation_tone`, `voice_modify`, and sentence/word subtitle timestamps.

Selector-style output aliases such as `mp3_44100_128` are parsed as MP3,
44.1 kHz, 128 kbps unless explicit `sample_rate` or `bitrate` values override
them. When status `2` confirms synthesis but local decoding or writing fails,
the failure result still carries the estimated cost, trace ID, usage, and
`resubmit_safe=false`. A POST timeout has unknown charge state and must not be
blindly retried.

## Approval Workflow

Both tools call paid external APIs. Before production:

1. Unless a cost-calculation opt-out is recorded, confirm the current MiniMax console price; direct-API pricing must not reuse a fal.ai estimate.
2. Generate one short video sample and one 10–15 second voice sample after creative approval (and cost approval only when the project has not opted out).
3. Approve motion consistency, voice identity, pronunciation, and pace.
4. After sample approval, continue in rolling micro-batches rather than generating the full remainder at once.

### Rolling micro-batches (mandatory)

Sample approval never authorizes a one-shot full-video batch. For paid MiniMax
video and TTS production:

1. Generate one story segment at a time. Default to one hero/dialogue clip per
   batch; at most two closely related clips or one short voice cluster.
2. Record task IDs, result paths, and provider-returned charge-state fields in
   the project checkpoint before starting the next segment. A project may opt
   out of cost calculations, but it must never waive task-ID persistence.
3. Review character identity, costume, palette, motion coherence, lip/audio
   timing, and continuity against the immediately preceding approved segment.
4. If the segment fails, regenerate only that segment. Do not rerun successful
   neighboring clips and do not submit the next segment while a paid task has
   unknown charge state.
5. Stop the rolling run on character/style drift, repeated provider failure,
   an unknown/anomalous provider charge state, or any decision that would switch provider,
   model family, runtime, or approved visual direction.

For a long film, organize micro-batches by contiguous story beats (for example
端午盛景 → 黄昏独候 → 初遇误会 → 火把归途), but still submit and verify
the individual 6/10-second MiniMax tasks inside each beat one by one.

### Cost-calculation opt-out

When the project decision log records that the user does not want cost
calculations (as in `bian-cheng`), that preference is binding:

- Do not calculate aggregate estimates, compare quality/budget tiers, request
  budget approval, or present routine cost figures to the user.
- Do not delay a previously approved micro-batch merely to obtain another cost
  confirmation.
- Continue the rolling micro-batch rule and per-item checkpoints exactly as
  written above; a cost opt-out never authorizes a full-film batch.
- Preserve provider-returned `task_id`, `file_id`, `charge_state`,
  `potential_cost_usd`, and `cost_usd` only as internal safety/audit metadata.
  Do not use those fields to create forecasts or user-facing totals.
- Missing or hidden cost information must never be interpreted as “free.” On
  unknown charge state, stop and resume by task ID instead of resubmitting.

OpenMontage uses MiniMax's published pay-as-you-go rates by default:

| Model/spec | Published price |
|---|---:|
| Hailuo 2.3, 768P, 6s | $0.28/video |
| Hailuo 2.3, 768P, 10s | $0.56/video |
| Hailuo 2.3, 1080P, 6s | $0.49/video |
| Hailuo 2.3 Fast, 768P, 6s / 10s | $0.19 / $0.32 |
| Hailuo 2.3 Fast, 1080P, 6s | $0.33/video |
| Speech HD | $0.10/1,000 characters |
| Speech Turbo | $0.06/1,000 characters |

Reconfirm the pricing page before paid generation because rates can change. Optional negotiated/package overrides are:

```text
MINIMAX_VIDEO_COST_PER_CLIP_USD
MINIMAX_VIDEO_COST_PER_SECOND_USD
MINIMAX_TTS_COST_PER_1000_CHARS_USD
```

## Troubleshooting

- `MINIMAX_API_KEY not set`: add the key to `.env`; do not pass it in prompts or tool inputs.
- HTTP succeeds but the tool reports an API error: inspect `base_resp.status_code/status_msg` for authentication, balance, rate-limit, or parameter errors.
- Video remains queued: keep the recommended polling interval and increase `timeout_seconds`; do not create duplicate paid tasks while the first task is still active.
- Image-to-video asks for `FAL_KEY`: the selector is using an old path; direct MiniMax must receive `reference_image_path` unchanged.
- TTS writes no audio: verify `data.audio` exists and `output_format` is `hex`.
- Wrong pronunciation: add MiniMax `pronunciation_tone` entries and regenerate only a short sample first.

## Official References

- https://platform.minimax.io/docs/api-reference/video-generation-t2v
- https://platform.minimax.io/docs/api-reference/video-generation-i2v
- https://platform.minimax.io/docs/api-reference/video-generation-query
- https://platform.minimax.io/docs/api-reference/file-management-retrieve
- https://platform.minimax.io/docs/api-reference/speech-t2a-http
- https://platform.minimax.io/docs/guides/pricing-paygo
