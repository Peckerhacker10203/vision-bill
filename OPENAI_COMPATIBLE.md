# OpenAI-compatible LLM backends

Vision Bill can send text and image requests to an OpenAI-compatible API, including
llama.cpp. Keep the endpoint local for local receipt processing. Ollama remains the
default. OpenAI compatibility covers inference; model loading and unloading need
an additional backend-specific contract.

## Configure a dedicated llama.cpp router

Use a llama.cpp build exposing the router management API described in its
[server guide](https://github.com/ggml-org/llama.cpp/blob/1ef8c5d68495d69b065d28e780264ee01595b6b4/tools/server/README.md#using-multiple-models).
A stock single-model server or an arbitrary OpenAI-compatible server does not
provide this contract merely by accepting chat completions.

Deploy a dedicated Vision Bill router. Start it in router mode without selecting
one model on the server command line; configure local GGUF files through its model
directory or presets. Keep autoload enabled and set every preset's
`load-on-startup=false`. Model discovery and readiness checks must not eagerly
load weights. `--models-max=1` alone does not coordinate the separate OCR service.

Replace the active LLM values in your private configuration, rather than appending
a second conflicting definition:

```dotenv
LLM__PROVIDER=openai
LLM__HOST=http://host.docker.internal:8083/v1
LLM__API_KEY=none
LLM__MODEL_NAME=receipt-formatter
LLM__MODEL_LIFECYCLE=llamacpp
OCR__ENABLED=true
OCR__MANAGE_MODEL_LIFECYCLE=true
```

The URL must reach your deliberately deployed router. The example uses the Docker
host gateway; a host process can use `http://127.0.0.1:8083/v1`. A host router must
listen on an interface reachable from the app container and remain on a trusted
private network. `none` is a placeholder for a router without authentication;
otherwise use its configured API key in the ignored private configuration.
The same Bearer key authenticates inference and lifecycle requests.

Replace `receipt-formatter` with its exact canonical `id` from the router's
`GET /models` response. Managed release does not resolve aliases. OCR formatting
can use a text-only model. Direct vision or an explicitly configured vision
fallback needs an image-capable model. For older servers with absent capability
metadata, `LLM__VISION_MODEL_IDS=["receipt-vision"]` can declare known image support;
it never overrides explicit contrary metadata. Output/context limits still apply;
this setting does not give a text model vision support.

The normal output cap remains 8192 tokens. The fixed native OCR 4096-token cap,
2,000,000-pixel region limit, orientation/unwarping and CUDA/ROCm profile are
unchanged. The lifecycle API is independent of the GPU backend selected when
deploying llama.cpp.

## Sequential GPU handoff

With managed OCR enabled, one internal owner coordinates the complete job:

1. Park any abandoned OCR worker and release the job's previously recorded LLM IDs.
2. Load native OCR on demand and process every receipt photo.
3. Park OCR and confirm that its native worker has exited.
4. Send OpenAI-compatible requests; router autoload loads the requested LLM.
   Formatting, repairs and classification share the same owner.
5. Request that LLM's unload and wait for its canonical router status to become
   `unloaded` before another owner can load OCR.

The control protocol is `GET /models` and
`POST /models/unload {"model":"canonical-id"}`. A successful POST only schedules
termination. Vision Bill polls until the exact model reports
`status.value="unloaded"`; sleeping, missing IDs, unknown/duplicate states or an
empty model list do not confirm release. Only the journal's exact owned model IDs
are released. Other discovered models are never swept.

Release has at most 30 seconds within the existing absolute attempt deadline.
An unconfirmed/failed/cancelled request keeps durable ownership unresolved and
blocks subsequent OCR. See [recovery](paddleocr-service/README.md#internal-model-ownership).

Every app worker and supported standalone client must use matching provider,
lifecycle mode and endpoint spelling plus the same underlying Linux
`OCR__COORDINATION_PATH` file and adjacent journal. The dedicated router must not
receive model-loading requests from outside those owners. This coordinates Vision
Bill's models and does not reserve the physical GPU; other GPU applications can
continue running.

Use the final inference URL without redirects. Lifecycle and managed inference
requests refuse redirects and ignore environment proxy variables. A host ending
in `/v1` uses sibling management routes; for example
`http://host/llama/v1` maps to `http://host/llama/models/unload`. Userinfo,
query/fragment components and ambiguous encoded/dot paths are rejected.

## Other providers, changes and verification

`LLM__MODEL_LIFECYCLE=none` is the default for OpenAI-compatible providers without
an implemented unload API. Pure LLM operation remains supported. Managed native
OCR refuses unsupported unloading. Selecting `llamacpp` alone does not enable
automatic unloading while production managed OCR is off. An explicitly unmanaged
CPU/replay or separate-resource policy makes no shared-GPU handoff claim and
cannot activate the shipped parked native sidecar without ownership.
Ollama uses its own lifecycle and ignores a retained OpenAI-only lifecycle option.

Drain and stop all owners before changing provider, endpoint or lifecycle mode.
Reconcile outstanding requests and owned models on the **old endpoint** first;
the existing journal binds its endpoint identity. Only after confirmed release,
with all owners stopped, archive the old lock/journal as described in recovery,
change configuration and restart. Lifecycle configuration requires a restart.

Build/install and ordinary service startup do not run receipt comparisons,
historical colleague/Gemma benchmarks or live inference tests. After building,
test your own uploads through the web UI. Software CI uses CPU doubles; these
contracts do not certify a real llama.cpp deployment or CUDA/ROCm hardware.
The context-budget diagnostic remains Ollama-only; the app and standalone
benchmark use the configured provider.
