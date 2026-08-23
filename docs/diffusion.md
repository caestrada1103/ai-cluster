# Diffusion (Image & Video Generation)

Optional service: [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
`sd-server`, run from upstream Docker images. It is not part of the core LLM
cluster — start it separately when you want image or video generation
alongside chat.

## APIs

One port, three surfaces:

| Endpoint | Style | Notes |
|---|---|---|
| `POST /v1/images/generations` | OpenAI-compatible | Sync, returns `b64_json` |
| `POST /sdapi/v1/txt2img`, `/img2img` | AUTOMATIC1111-compatible | Sync |
| `POST /sdcpp/v1/img_gen`, `/vid_gen` | Native, async | Returns `202` + `{id, poll_url}`; poll `GET /sdcpp/v1/jobs/{id}` |

**Video only exists on the native `/sdcpp/v1/vid_gen` API.** A completed
native job's `result` has `b64_json`, `mime_type`, and `frame_count`.

## Picking an image tag

One image tag per GPU vendor — pick the one matching your hardware.

| Tag | GPU | Notes |
|---|---|---|
| `master-cuda-spark` | DGX Spark (GB10, sm_121) | Default in `scripts/run-diffusion.sh` |
| `master-cuda` | RTX 3080 and other NVIDIA | |
| `master-vulkan` | AMD RX 9060XT | |
| `master-sycl` | Intel | |

All four are published under `ghcr.io/leejet/stable-diffusion.cpp:<tag>`.
Override with `--tag` on the CLI or `SD_IMAGE_TAG` in `.env`.

## Starting the server

```bash
cp .env.example .env   # if you haven't already; set SD_MODEL or SD_EXTRA_ARGS

# Single-file image model (e.g. an SDXL checkpoint)
./scripts/run-diffusion.sh --model sd_xl_turbo_1.0.safetensors --tag master-cuda-spark

# RTX 3080
./scripts/run-diffusion.sh --model sd_xl_turbo_1.0.safetensors --tag master-cuda

# Follow logs / stop
./scripts/run-diffusion.sh --logs
./scripts/run-diffusion.sh --stop
```

`--name` picks the container name if you need more than one instance running
(each `sd-server` serves exactly one loaded model at a time).

`sd-server` loads one model per process, so images and video run as two
instances on two ports:

```bash
# images on 8090
./scripts/run-diffusion.sh --model RealVisXL_V5.0_fp16.safetensors --port 8090

# video on 8091
SD_EXTRA_ARGS="--diffusion-model /models/diffusion_models/Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf \
  --high-noise-diffusion-model /models/diffusion_models/Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf \
  --vae /models/vae/wan_2.1_vae.safetensors \
  --t5xxl /models/text_encoders/umt5-xxl-encoder-Q8_0.gguf --diffusion-fa" \
  ./scripts/run-diffusion.sh --name ai-sd-video --port 8091
```

Point the video Pipe's `base_url` Valve at the video instance (`http://127.0.0.1:8091`
by default).

The two instances need not share a host. To run the video generator on another
machine, set `SD_SERVER_HOST` to an address that machine's peers can reach, and
point the Pipe's `base_url` there:

```bash
SD_SERVER_HOST=10.0.0.2 SD_SERVER_PORT=8091 SD_EXTRA_ARGS="..." \
  ./scripts/run-diffusion.sh --name ai-sd-video
```

Because the model is loaded per process, this splits the workload across
machines; it does not pool memory for a single generation.

## Models

`scripts/fetch-diffusion-models.sh` downloads known-good sets:

```bash
./scripts/fetch-diffusion-models.sh --list
./scripts/fetch-diffusion-models.sh realvis        # photorealistic SDXL, 6.9 GB
./scripts/fetch-diffusion-models.sh wan22-i2v      # image-to-video, ~37 GB
```

Files land in `SD_MODELS_DIR` (default `./models/diffusion`, mounted read-only
at `/models`). Generated output that sd-server writes server-side lands in
`SD_OUTPUT_DIR` (default `./data/diffusion`, mounted at `/output`). Both
directories are gitignored.

- **Single-file image models** (most SD/SDXL checkpoints): pass `--model
  <file>` (relative to `SD_MODELS_DIR`), or set `SD_MODEL` in `.env`.
- **Video models** (multi-file: diffusion model + VAE + text encoder) don't
  fit the single `--model` flag. Point `SD_EXTRA_ARGS` at them directly:

```bash
SD_EXTRA_ARGS="--diffusion-model /models/wan2.1-t2v-14b.gguf --vae /models/wan_vae.safetensors --t5xxl /models/t5xxl_fp8.safetensors"
```

Leave `SD_MODEL` unset when using `SD_EXTRA_ARGS` this way — `run-diffusion.sh`
only requires one of the two.

## Wiring Open-WebUI: images

Open-WebUI has a native `openai` image engine — no custom code needed.

**Env vars** (Open-WebUI container):

```
ENABLE_IMAGE_GENERATION=true
IMAGE_GENERATION_ENGINE=openai
IMAGES_OPENAI_API_BASE_URL=http://127.0.0.1:8090/v1
IMAGES_OPENAI_API_KEY=anything-non-empty
IMAGE_SIZE=1024x1024
IMAGE_STEPS=20
IMAGE_GENERATION_MODEL=sd_xl_turbo_1.0
```

`sd-server` ignores the API key value — any non-empty string works. The live
`ai-open-webui` container runs `--network host`, so `127.0.0.1:8090` reaches
`sd-server` directly on this host.

Same settings are also reachable in the UI: **Admin Panel → Settings →
Images**.

## Wiring Open-WebUI: video

Open-WebUI has no native video generation (verified in source; upstream
discussion [#18546](https://github.com/open-webui/open-webui/discussions/18546)
is still open). `diffusion/openwebui_video_pipe.py` is a Pipe function that
works around this: it appears as a selectable model, **"Video
(stable-diffusion.cpp)"**, in the chat model dropdown, calls
`/sdcpp/v1/vid_gen`, polls the job, and returns a
`<video controls src="data:video/webm;base64,...">` block — Open-WebUI's
`HTMLToken.svelte` renders that as a real inline player.

**Install:**

1. Open-WebUI → **Workspace → Functions → +** (new function).
2. Paste the contents of `diffusion/openwebui_video_pipe.py`.
3. Save, then enable it.
4. Select model **"Video (stable-diffusion.cpp)"** in the chat dropdown.

**Configure via the function's Valves** (gear icon on the function):

| Valve | Default | Purpose |
|---|---|---|
| `base_url` | `http://127.0.0.1:8091` | `sd-server` base URL (the video instance) |
| `width` | `720` | Output width |
| `height` | `1280` | Output height |
| `video_frames` | `97` | Frame count, rounded to 4n+1 |
| `fps` | `16` | Output frames per second |
| `sample_steps` | `10` | Denoising steps — the main speed/quality dial |
| `high_noise_sample_steps` | `8` | Steps for the high-noise expert (Wan2.2 A14B) |
| `txt_cfg` | `3.5` | Prompt guidance scale |
| `flow_shift` | `3.0` | Flow shift |
| `lora` | (empty) | LoRA tags appended to the prompt |
| `strength` | `0.75` | `init_image` denoise strength |
| `negative_prompt` | (empty) | Negative prompt |
| `output_format` | `webm` | `webm`, `webp`, or `avi` |
| `timeout_s` | `3600` | Give up on a job after this long |
| `poll_interval_s` | `3.0` | Job poll interval |
| `save_dir` | (empty) | Also write the file to this path on the server, if set |
| `embed_inline` | `true` | Fall back to a base64 data URL when the file cannot be registered |

## Saving generated media to the client PC

Both images and video are delivered as real Open-WebUI files, so the browser
downloads them to whichever machine is viewing the page — not to the host
running `sd-server`.

- **Images** — click the image in chat, then use the download control in the
  lightbox (or right-click → Save image).
- **Video** — the Pipe returns an inline `<video>` player plus a **Download**
  link. The link hits `/api/v1/files/{id}/content?attachment=true`, which
  sets `Content-Disposition: attachment`.

`save_dir` is separate: it writes a server-side copy under `SD_OUTPUT_DIR` on
the machine running the Pipe. Use it for archiving, not for delivering files
to the viewer.

Output is WebM. Browsers play it natively, but most social platforms prefer
MP4:

```bash
ffmpeg -i clip.webm -c:v libx264 -pix_fmt yuv420p clip.mp4
```

`video_frames` is rounded **down** to the nearest 4n+1 before it's sent to
`sd-server` (e.g. a Valve of `100` sends `97`) — that's a stable-diffusion.cpp
encoding requirement, not a bug.

## Going faster

Generation cost scales with parameters x steps x pixels x frames. A *bigger*
model is slower, not faster — steps are the dial that matters.

`sample_steps`/`high_noise_sample_steps` default to `10`/`8`, matching
stable-diffusion.cpp's reference config for Wan2.2. Lower them for speed,
raise them for quality.

Below ~8 steps, use a distillation LoRA instead of just cutting steps:

```bash
./scripts/fetch-diffusion-models.sh wan22-lora
```

Then set the Pipe's `lora` Valve, and `sample_steps`/`high_noise_sample_steps`
to `4`:

```
<lora:wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022:1><lora:|high_noise|wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022:1>
```

The `|high_noise|` prefix targets the high-noise expert — Wan2.2-A14B is a
mixture of experts and each half needs its own LoRA. LoRAs load from
`SD_MODELS_DIR/loras`.

Other levers, roughly in order of payoff: fewer `video_frames`, lower
`width`/`height`, then a smaller model (Wan2.2-TI2V-5B is ~3x lighter than
A14B at some cost in fidelity).

## Performance note

On a DGX Spark, expect video generation to take **many minutes per clip**.
GB10 has roughly 273 GB/s of memory bandwidth — far below a discrete NVIDIA
GPU — and diffusion sampling is iterative (dozens of sequential denoising
steps over the whole frame stack), so the process is bandwidth-bound the
whole way through. The Spark's advantage here is unified-memory **capacity**
(fitting large video models at all), not generation speed. Set the video
Pipe's `timeout_s` accordingly, and expect image generation (`txt2img`, far
fewer steps over a single frame) to be much faster than video on the same
hardware.

## Troubleshooting

- **`run-diffusion.sh` exits with "No model."** — set `SD_MODEL`/`--model`
  for a single-file checkpoint, or `SD_EXTRA_ARGS` for a multi-file video
  model (see "Models" above).
- **Container starts then exits immediately** — `docker logs ai-sd-server`
  (or `--logs`); usually a missing/misnamed model file under `SD_MODELS_DIR`.
- **No GPU found inside the container** — confirm the tag matches your
  vendor (table above); `*-cuda*` tags need `--gpus all` (NVIDIA Container
  Toolkit installed), `*-vulkan`/`*-sycl` need `/dev/dri` passed through,
  which `run-diffusion.sh` does automatically based on `--tag`.
- **Open-WebUI image generation returns an error** — check
  `IMAGES_OPENAI_API_BASE_URL` includes `/v1`, and that `sd-server` is
  actually reachable from the Open-WebUI container (same host + `--network
  host` for the shipped `ai-open-webui` container; otherwise use the compose
  service name over `ai-cluster-net`, not `127.0.0.1`).
- **Video Pipe times out** — raise `timeout_s`; see the performance note
  above, especially on a DGX Spark.
- **Video Pipe returns "sd-server rejected the job (4xx)"** — the response
  body (truncated in the reply) usually names the bad field; check
  `width`/`height`/`video_frames` are within the model's supported range.
- **Poll returns 404/410** — the job ID is gone, usually because `sd-server`
  restarted mid-job; resubmit.
- **`filesystem error: ... Operation not permitted [./proc/...]`** —
  `sd-server` scans the working directory for LoRAs. `run-diffusion.sh` sets
  `-w /models` and `--lora-model-dir /models/loras` to keep that scan off
  `/proc`; a hand-rolled `docker run` without them hits this on every
  `vid_gen` and `/sdcpp/v1/capabilities` call.
- **Video doesn't play inline in Open-WebUI** — confirm `embed_inline` is
  `true` and `output_format` is `webm` (broadest browser support; `avi` may
  not play in-browser even though `sd-server` will happily generate it).
