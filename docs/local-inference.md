# Local Inference

Both Capture and Wiki can run entirely on local inference. This page documents what that actually looks like on commodity hardware — real numbers, real constraints, no benchmarketing.

## Why Local

Capture classifies conversations. Wiki compiles articles. Both are structured tasks with clear success criteria — the LLM reads input and produces categorized output. These don't need frontier models. A quantized open-weight model running on a mini PC handles them fine.

Running locally means:
- Zero marginal cost per conversation processed
- Zero marginal cost per wiki article compiled
- No API rate limits
- No data leaving your machine
- No subscription that stops working when you cancel

The tradeoff is speed and quality ceiling. Local models are slower and make more mistakes on nuanced reasoning than commercial APIs. For classification and compilation, the error rate is acceptable. For open-ended analysis, it's not — but that's not what these tools do.

## Hardware

This system runs on a **Beelink EQI12** — a $300 mini PC.

| Spec | Value |
|------|-------|
| CPU | Intel i5-1235U (2P + 8E cores, 12 logical) |
| RAM | 31 GiB DDR4 |
| GPU | None (integrated only, not used) |
| Storage | NVMe |

Not a workstation. Not a GPU rig. A small box that sits on a shelf and costs less than three months of a managed knowledge service subscription.

## Runtime

**llama-server** from [llama.cpp](https://github.com/ggerganov/llama.cpp), CPU-only.

Key configuration:
```
--threads 8              # Physical cores only
--ctx-size 8192          # Sufficient for classification/compilation
--cache-type-k q4_0      # Quantized K-cache for memory savings
--cache-type-v q8_0      # Higher precision V-cache
--mlock                  # Lock model in RAM
--flash-attn on          # CPU efficiency
--parallel 1             # One request at a time
```

## Model

**Gemma-4 26B** (GGUF quantization). Any model that fits in RAM and speaks the OpenAI API format works. Gemma-4 26B is what's been tested in production.

## Real Performance

Daily benchmarks from production operation (April 2026):

| Metric | Value |
|--------|-------|
| Generation speed | ~10 tokens/sec |
| Prompt processing | ~7.5 tokens/sec |
| Time to first token | ~900ms |
| RAM usage | ~20-25 GiB |

At 10 tokens/sec, classifying a conversation takes a few seconds. Compiling a wiki article from a research brief takes 10-30 seconds depending on length. A nightly batch of 20 conversations processes in under 5 minutes.

## Memory Pressure

The model consumes most of the 31 GiB available. With the OS and other services, free RAM drops to 5-6 GiB with 9+ GiB in swap. This causes gradual speed degradation as swap pressure increases.

This is the honest picture of running a 26B model on 31 GiB of RAM. It works. It's slow. It gets slower over time as memory fills up. If you're planning something similar, either use a smaller model or get more RAM.

## On the Trajectory

Running a 26B model on a $300 CPU-only mini PC for production tasks was not possible two weeks before this system went live. The model and quantization tooling are both recent developments.

The trend in open-weight models is toward more capability on less hardware. The model that replaces Gemma-4 26B on this same box in six months will be faster and more capable. The box itself could be upgraded for a few hundred dollars.

The durable part isn't the specific model or hardware — it's the pattern: **run background knowledge tasks locally, save API spend for work that faces humans.** That pattern survives hardware upgrades, model releases, and capability improvements. The mini PC is a consumable. The architecture is not.

## API Fallback

If local inference is unavailable (server down, box off, model swapped), both Capture and Wiki fall back to OpenRouter or any configured API endpoint. The scripts try local first, fall back silently, and log which backend answered.

This means you can start with API-only (no local hardware), add local inference later when you want to reduce costs, and the scripts don't change. The LLM endpoint is configuration, not code.

## What Doesn't Work Locally

**Interactive conversation.** At 10 tokens/sec, generating a long response takes 30+ seconds. Fine for batch processing. Unacceptable for real-time use.

**Complex multi-step reasoning.** A 26B quantized model on CPU makes more mistakes than a commercial model on nuanced tasks. Classification and compilation have clear success criteria where errors are visible and correctable. Open-ended analysis does not.

**Concurrent requests.** One request at a time with `--parallel 1`. If the wiki compile step and a conversation classification run simultaneously, one waits. In practice this rarely conflicts because they run on different cron schedules.
