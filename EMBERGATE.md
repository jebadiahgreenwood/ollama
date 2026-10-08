# EmberGate fork of Ollama

Branch `embergate` starts at upstream **v0.35.0** (`cc40693`), the exact engine EmberGate runs (`ollama/ollama:0.35.0`).
`main` tracks upstream untouched.

**Why:** the kid (the EmberGate seedling, a Qwen3.5-9B fine-tune) has a trained MTP head. Upstream Ollama only uses MTP
heads on Apple/MLX. On CUDA we turn on llama.cpp's `draft-mtp` speculative decoding through `LLAMA_ARG_SPEC_*` env vars
that the bundled llama-server inherits. That works (1.8–1.9× generation, same choices), but it is instance-wide and blind:
a model without an MTP head fails to load, and the draft/accept counts only exist in the server log.

**What this fork is for** (none of it done yet):
- per-model speculative settings (Modelfile `PARAMETER`), not one env switch for the whole server
- draft / accepted counts in the `/api/chat` response, so acceptance becomes a live per-turn signal
  (it rises when the kid repeats itself: AUROC .68 for repeated tool calls)
- graceful fallback when a model has no MTP head
- whatever the kid proposes once it can read this code

**How it's served:** EmberGate host compose `services/ollama-kid/` builds `FROM ${OLLAMA_BASE}`; build this branch's
image and pass its tag as `OLLAMA_BASE`. Current service: `ollama-kid-mtp` (harbor backend `ollama-kid`).
