# Why AI Cluster?

## The Problem: Your GPU Is Sitting Idle, and the Model Doesn't Fit Anyway

Most people don't have a single high-end AI accelerator — they have a
gaming PC (or two) with a consumer NVIDIA or AMD card in the 8–16 GB VRAM
range, often sitting idle outside of gaming sessions. A full-precision LLM
checkpoint routinely needs far more VRAM than that, so the card goes
unused for inference even though it's perfectly capable of running a
right-sized, quantized model.

### Key Challenges for Individuals and Small Teams:
1.  **VRAM Limits:** A full-precision (FP32/FP16) checkpoint of a useful
    model often doesn't fit an 8–16 GB consumer card.
2.  **High Costs:** Enterprise-grade GPUs (like A100s or H100s) are
    prohibitively expensive for most developers and startups.
3.  **Complexity:** Setting up quantized inference and/or a distributed
    setup manually is technically daunting — picking the right GGUF quant,
    wiring up GPU drivers, exposing an API.
4.  **Hardware Fragmentation:** Users often have a mix of different GPUs
    (e.g., an older NVIDIA card on one machine and a newer AMD card on
    another), making unified utilization difficult.

## The Solution: AI Cluster

**AI Cluster** lets you run inference on the consumer GPU(s) you already
own — NVIDIA or AMD, including a card that would otherwise sit idle — using
a **quantized GGUF model** via the built-in llama.cpp engine, sized to fit
in the VRAM you actually have. A Python coordinator gives you a single
OpenAI-compatible API in front of one or more Rust workers, so you don't
have to hand-roll the GPU driver, quantization, and serving plumbing
yourself.

### How It Works (Simplified)
Imagine your GPU is a small bookshelf, and the AI Model is a book that's
too thick to fit on it at full size.
- **Without AI Cluster:** You can't fit the book on the shelf, so it stays
  in the box.
- **With AI Cluster:** You get an abridged edition of the book (a quantized
  GGUF file — same story, smaller footprint) that fits comfortably on your
  shelf. The "Coordinator" hands you a single front door (an OpenAI-style
  API) regardless of which shelf (worker/GPU) is holding the book. If the
  book is too thick even in abridged form, splitting it across a few
  shelves (multi-GPU model split) is the direction this project is headed
  next — not yet available through the worker today.

### Key Benefits
*   **Fit Bigger Models in Small VRAM:** A quantized GGUF checkpoint
    (Q4_K_M/Q5_K_M/Q8_0/…) can shrink a model enough to run inference on a
    single ~8–16 GB consumer card that couldn't hold the full-precision
    weights.
*   **Cost Efficiency:** Use the hardware you already own — no need to buy
    an A100, just point AI Cluster at your gaming PC's idle GPU.
*   **Mixed Hardware Support:** NVIDIA and AMD cards both work (CUDA/ROCm/
    Vulkan), including in the same cluster, so an older card and a newer
    one from different vendors can both serve requests.
*   **One API, Two Engines:** An OpenAI-compatible REST API in front of
    either the primary llama.cpp/GGUF engine (quantized, recommended) or
    the experimental Burn engine (FP32 reference, single GPU).

Splitting one model across several consumer GPUs — the other half of "make
big models fit" — rides on parallelism work that exists in the codebase
(`worker/src/parallelism.rs` for the Burn engine, upstream llama.cpp's
native multi-GPU split for GGUF) but isn't wired into the inference path
yet; see [architecture.md](architecture.md#parallelism-strategies) for
current status. Today, "does it fit" is decided per-GPU, by quantization.

### Who is this for?
*   **Researchers & Students:** Experiment with capable open models without
    university-scale budgets.
*   **Startups:** Prototype and deploy private AI services without relying
    on expensive cloud APIs or dedicated enterprise hardware.
*   **Hobbyists:** Put a spare or idle gaming GPU to productive use.

AI Cluster bridges the gap between the consumer hardware you already have
and running real LLMs on it.
