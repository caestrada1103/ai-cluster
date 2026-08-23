"""Open-WebUI Pipe: video generation via stable-diffusion.cpp.

Install by pasting this file into Open-WebUI > Workspace > Functions.
Setup and troubleshooting: docs/diffusion.md
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

# sd-server encodes 4n+1 frames; other counts are rounded down.
FRAME_STEP = 4


def _round_frames(n: int) -> int:
    """Round down to the nearest 4n+1."""
    return max(FRAME_STEP + 1, ((n - 1) // FRAME_STEP) * FRAME_STEP + 1)


class Pipe:
    class Valves(BaseModel):
        base_url: str = Field(
            default="http://127.0.0.1:8091",
            description="sd-server base URL. Point at the video instance, local or remote.",
        )
        width: int = Field(default=720, description="Output width")
        height: int = Field(default=1280, description="Output height")
        video_frames: int = Field(default=97, description="Frame count, rounded to 4n+1")
        fps: int = Field(default=16, description="Output frames per second")
        sample_steps: int = Field(default=10, description="Denoising steps")
        high_noise_sample_steps: int = Field(
            default=8, description="Denoising steps for the high-noise expert (Wan2.2 A14B)"
        )
        txt_cfg: float = Field(default=3.5, description="Prompt guidance scale")
        flow_shift: float = Field(default=3.0, description="Flow shift")
        lora: str = Field(
            default="",
            description="LoRA tags appended to the prompt, e.g. "
            "<lora:name_low:1><lora:|high_noise|name_high:1>",
        )
        strength: float = Field(default=0.75, description="init_image denoise strength")
        negative_prompt: str = Field(default="", description="Negative prompt")
        output_format: str = Field(default="webm", description="webm, webp or avi")
        timeout_s: int = Field(default=3600, description="Give up on a job after this long")
        poll_interval_s: float = Field(default=3.0, description="Job poll interval")
        save_dir: str = Field(
            default="", description="Also write the file to this path on the server, if set"
        )
        embed_inline: bool = Field(
            default=True,
            description="Fall back to a base64 data URL when the file cannot be registered",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    def pipes(self) -> List[Dict[str, str]]:
        return [{"id": "sdcpp-video", "name": "Video (stable-diffusion.cpp)"}]

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        prompt, init_image = _last_user_turn(body.get("messages") or [])
        if not prompt and not init_image:
            return "Send a prompt, or attach an image, to generate a video."

        v = self.valves
        sample_params = {
            "sample_steps": v.sample_steps,
            "flow_shift": v.flow_shift,
            "guidance": {"txt_cfg": v.txt_cfg},
        }
        payload: Dict[str, Any] = {
            "prompt": prompt + v.lora,
            "negative_prompt": v.negative_prompt,
            "width": v.width,
            "height": v.height,
            "video_frames": _round_frames(v.video_frames),
            "fps": v.fps,
            "strength": v.strength,
            "seed": -1,
            "output_format": v.output_format,
            "sample_params": sample_params,
            "high_noise_sample_params": {
                **sample_params,
                "sample_steps": v.high_noise_sample_steps,
            },
        }
        if init_image:
            payload["init_image"] = init_image

        async with httpx.AsyncClient(base_url=v.base_url, timeout=60.0) as client:
            try:
                job = await _submit(client, payload)
            except httpx.HTTPStatusError as exc:
                return (
                    f"sd-server rejected the job ({exc.response.status_code}): "
                    f"{exc.response.text[:400]}"
                )
            except httpx.HTTPError as exc:
                return f"Cannot reach sd-server at {v.base_url}: {exc}"

            result = await self._await_job(client, job["id"], __event_emitter__)

        if isinstance(result, str):
            return result

        data = base64.b64decode(result["b64_json"])
        mime = result.get("mime_type", "video/webm")
        note = ""
        if v.save_dir:
            note = _save(Path(v.save_dir), job["id"], v.output_format, data)

        secs = result.get("frame_count", 0) / max(result.get("fps", 1), 1)
        caption = f"{result.get('frame_count')} frames · {secs:.1f}s · {len(data) / 1e6:.1f} MB"

        url = await _register_file(
            data, f"{job['id']}.{v.output_format}", mime, (__user__ or {}).get("id")
        )
        if url:
            return (
                f'<video controls src="{url}"></video>\n\n'
                f"[Download]({url}?attachment=true) · {caption}{note}"
            )
        if not v.embed_inline:
            return note or "Generated, but the file could not be registered."

        b64 = base64.b64encode(data).decode()
        return f'<video controls src="data:{mime};base64,{b64}"></video>\n\n{caption}{note}'

    async def _await_job(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        emit: Optional[Callable[[dict], Awaitable[None]]],
    ) -> Any:
        """Poll until the job leaves a running state, or the timeout expires."""
        deadline = time.monotonic() + self.valves.timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(self.valves.poll_interval_s)
            try:
                resp = await client.get(f"/sdcpp/v1/jobs/{job_id}")
            except httpx.HTTPError as exc:
                return f"Lost contact with sd-server while polling: {exc}"
            if resp.status_code in (404, 410):
                return f"Job {job_id} is gone ({resp.status_code}); sd-server may have restarted."
            resp.raise_for_status()
            job = resp.json()
            status = job.get("status")

            if status == "completed":
                await _status(emit, "Done", done=True)
                return job["result"]
            if status in ("failed", "cancelled"):
                err = (job.get("error") or {}).get("message", status)
                await _status(emit, f"Failed: {err}", done=True)
                return f"Generation {status}: {err}"

            elapsed = int(time.monotonic() - (deadline - self.valves.timeout_s))
            queued = job.get("queue_position") or 0
            where = f"queued at {queued}" if status == "queued" else "generating"
            await _status(emit, f"{where} · {elapsed}s")

        await _status(emit, "Timed out", done=True)
        return (
            f"Timed out after {self.valves.timeout_s}s. "
            "The job may still finish; check sd-server."
        )


async def _submit(client: httpx.AsyncClient, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = await client.post("/sdcpp/v1/vid_gen", json=payload)
    resp.raise_for_status()
    return resp.json()


async def _status(
    emit: Optional[Callable[[dict], Awaitable[None]]], text: str, done: bool = False
) -> None:
    if emit:
        await emit({"type": "status", "data": {"description": text, "done": done}})


async def _register_file(
    data: bytes, filename: str, mime: str, user_id: Optional[str]
) -> Optional[str]:
    """Store the clip as an Open-WebUI file and return its URL, or None."""
    if not user_id:
        return None
    try:
        from open_webui.config import UPLOAD_DIR
        from open_webui.models.files import FileForm, Files
    except ImportError:
        return None
    try:
        file_id = str(uuid.uuid4())
        path = Path(UPLOAD_DIR) / f"{file_id}_{filename}"
        path.write_bytes(data)
        record = await Files.insert_new_file(
            user_id,
            FileForm(
                id=file_id,
                hash=hashlib.sha256(data).hexdigest(),
                filename=filename,
                path=str(path),
                data={},
                meta={"name": filename, "content_type": mime, "size": len(data)},
            ),
        )
        return f"/api/v1/files/{record.id}/content" if record else None
    except Exception:
        return None


def _save(directory: Path, job_id: str, fmt: str, data: bytes) -> str:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{job_id}.{fmt}"
        path.write_bytes(data)
        return f"\n\nSaved to `{path}`"
    except OSError as exc:
        return f"\n\nCould not save to `{directory}`: {exc}"


def _last_user_turn(messages: List[Dict[str, Any]]) -> tuple[str, Optional[str]]:
    """Return (text, init_image) from the newest user message."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip(), None
        text_parts: List[str] = []
        image: Optional[str] = None
        for part in content or []:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url" and image is None:
                image = (part.get("image_url") or {}).get("url")
        return " ".join(text_parts).strip(), image
    return "", None
