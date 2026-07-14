# Unlimited-OCR runtime workspace

This directory contains the experimental Unlimited-OCR runtime workspace for
vllm-rbln. Backend choices are controlled through one CLI surface:
`--rbln-components`. Components not listed there run on CPU.

## Current runtime shape

1. Load the official HF remote-code model as the semantic source of truth.
2. Prepare one-image OCR inputs on CPU.
3. Optionally run selected components on RBLN/NPU via `--rbln-components`.
4. Compile `sam_model` on NPU by default when requested (included in `all`).
5. Run the language decoder on CPU by default, or on RBLN/NPU when
   `language_decode` is requested. Language **prefill** stays eager.

Selectable RBLN components:

- `sam_model`
- `vision_model`
- `projector`
- `embed_tokens`
- `language_decode`

Aliases:

- `vision_frontend` = `vision_model,projector`
- `quality_safe` = `embed_tokens`
- `all` = `sam_model,vision_model,projector,embed_tokens,language_decode`

## File layout

- `runtime.py`: runtime facade, backend summary, execution plan, and generation
  paths.
- `infer.py`: canonical CLI entry point.
- `processor.py`: image/PDF/text preprocessing and image-mode token packaging.
- `rbln_compile.py`: RBLN compile/cache dispatch helpers.
- `experiments/`: non-production language/RBLN research helpers.
- `docs/`: reference analysis notes moved out of the executable runtime path.

## Run

CPU default:

```bash
.venv/bin/python -m examples.experimental.unlimited_ocr.infer \
  --model baidu/Unlimited-OCR \
  --image /path/to/image.png \
  --image-mode gundam \
  --max-length 32768
```

Full NPU path (`all`, including SAM; language prefill stays eager):

```bash
.venv/bin/python -m examples.experimental.unlimited_ocr.infer \
  --model baidu/Unlimited-OCR \
  --image examples/experimental/unlimited_ocr/examples/ocr_test_multiline_ascii.png \
  --image-mode gundam \
  --rbln-components all \
  --max-length 32768 \
  --benchmark-runs 1 \
  --warmup-runs 0 \
  --output-json /tmp/uocr_all_rbln_128.json
```

Equivalent explicit form:

```bash
--rbln-components sam_model,vision_model,projector,embed_tokens,language_decode
```

## Important contracts


### `max_length`

The CLI follows the HF Unlimited-OCR naming and defaults to `--max-length 32768`.
Internally, runtime paths convert this total sequence limit to an effective new-token
budget with `max_length - prompt_length`.

### `embed_tokens`

When `embed_tokens` is compiled, the facade uses the compiled callable for prompt
embedding construction but deliberately keeps the original HF
`root_model.embed_tokens` module installed. This preserves
`model.embed_tokens.weight` in `hf_model.state_dict()` so the native PA-SWA
language runner can load all LLM weights.

### `language_decode`

`language_decode` selects the production-style PA-SWA cached NPU decoder runtime.
It does not generate an HF reference and does not run per-step eager replay.
Compiled decoder logits drive token selection, and compiled KV-cache outputs
drive the next decode step. Compile/fallback blockers are fail-closed instead of
silently reverting to CPU.

Current scope: batch=1, single-request, single-partition greedy decode. This is a
production-style runtime separation, not a multi-request serving scheduler.

### `sam_model`

`sam_model` is included in the `all` alias and compiles on NPU by default on the
v0.11 stack (rebel-compiler 0.11 compiled-vs-eager SAM parity is near-exact).
Omit `sam_model` from `--rbln-components` to keep SAM on CPU.

## Runtime metrics

Every CLI run prints a stage timing table and writes grouped rows to
`stage_timings` plus component rows to `component_timings` in the JSON output.
The JSON also includes `runtime_contract`, `backend_summary`, and the requested
RBLN component set.
