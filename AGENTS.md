# Repository guidance

## Scope

This repository provides the fine-grained recognition backend built on GroundingDINO. The main service logic is in `api.py` and the repository-root `utils.py`; the `groundingdino/` package contains the upstream model implementation.

## Architecture and runtime

- The FastAPI application is `api:app` and normally serves `/groundingdino/predict` on port 7579.
- Service startup loads multiple GPU, PyTorch, ONNX, face, landmark, logo, flag, flower, bird, car, and airplane models.
- Model weights and inference assets are external runtime data under `weights/` and `checkpoint/`.
- `CAPTION_API_URL` configures the optional caption dependency; do not hard-code new environment-specific service addresses.
- Preserve the upstream `LICENSE`, attribution, and relevant README material.

## External assets and security

- Never add model weights or generated inference assets to Git. This includes `weights/`, `checkpoint/`, and files ending in `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, `.npz`, or `.pkl`.
- Do not download, replace, rename, convert, or regenerate model assets unless explicitly requested.
- Never put API keys or tokens in source code, notebooks, comments, examples, or test files. Use environment variables.
- Treat model files as server-managed assets; local Mac checkouts may not be able to run full inference.

## Development commands

- Run lightweight syntax checks with `python3 -m py_compile api.py utils.py test_concurrent.py test_flag.py`.
- When the correct CUDA environment, compiled extension, dependencies, and weights are available, serve with `uvicorn api:app --host 0.0.0.0 --port 7579`.
- Do not run `test_flag.py` as a routine check: it writes generated class metadata under `weights/`.
- Do not run full model startup or GPU inference merely to validate an unrelated code change.

## Change rules

- Prefer focused changes in `api.py` and root `utils.py` for backend behavior.
- Modify the upstream `groundingdino/` implementation only when the task requires a model-level change; explain why the upstream code must change.
- Preserve request/response compatibility with `VL-api` unless both repositories are explicitly in scope.
- Avoid broad refactors across CUDA/C++ extension sources without an environment capable of rebuilding and testing them.
- Ask before adding or upgrading production ML dependencies; CUDA, PyTorch, ONNX Runtime, and compiled extensions are compatibility-sensitive.

## Verification

- Always run the lightweight syntax checks after relevant Python edits.
- For model or endpoint changes, state whether verification was syntax-only, CPU-only, or performed on the GPU server with real weights.
- Before handing off, inspect `git diff`, `git status --short`, and confirm no ignored model asset is staged.

## Git

- Work on a feature branch for non-trivial changes.
- Keep the Apache-2.0 license and upstream attribution intact.
- Do not commit, push, force-push, replace weights, or restart the deployed service unless explicitly requested.
