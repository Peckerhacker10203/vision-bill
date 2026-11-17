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
OCR__ENDPOINT=http://your-private-managed-ocr:8082
OCR__MANAGE_MODEL_LIFECYCLE=true
```

The LLM URL must reach your deliberately deployed router. Set the OCR endpoint to
your actual managed service; an independent OCR-only bundle must use
`OCR_SERVICE_MODE=managed` when paired with VisionBill. Combined bundles fix it
to managed. See [prepared installation](distribution/README.md). The example uses the
Docker host gateway; a host process can use `http://127.0.0.1:8083/v1`. A host router must
listen on an interface reachable from the app container and remain on a trusted
private network. `none` is a placeholder for a router without authentication;
otherwise use its configured API key in the ignored private configuration.
The same Bearer key authenticates inference and lifecycle requests.

Replace `receipt-formatter` with its exact canonical `id` from the router's
`GET /models` response. Managed release does not resolve aliases. OCR formatting
can use a text-only model and remains text/evidence-only on all ordinary failure,
repair and classification paths. Direct vision without OCR, or an explicit admin
direct-vision comparison, needs an image-capable model. For older servers with
absent capability
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
OCR waits before claim when unloading support or stable identities are missing;
no attempt is spent and coordination is never silently disabled. Selecting
`llamacpp` alone does not enable automatic unloading while production managed
OCR is off. An explicitly unmanaged
CPU/replay test policy makes no shared-GPU handoff claim. Do not use lifecycle
opt-out for a native shared pair; a cold unmanaged native pipeline waits before
claim and cannot activate the parked worker without ownership.
Ollama uses its own lifecycle and ignores a retained OpenAI-only lifecycle option.

Drain and stop all owners before changing provider, endpoint or lifecycle mode.
Reconcile outstanding requests and owned models on the **old endpoint** first;
the existing journal binds its endpoint identity. Only after confirmed release,
with all owners stopped, archive/reset the old handoff journal as described in
recovery. Keep the lockfile inode intact, change configuration and restart. Lifecycle configuration requires a restart.

Build/install and ordinary API startup do not run unsolicited comparisons or
historical colleague/Gemma benchmarks. Persisted explicit pending work can resume
under readiness/policy guards. Prepared native first setup separately runs full
required qualification; model provisioning uses pull/tags data checks only. After
installation, test your own uploads through the web UI. Software CI uses CPU
doubles; these contracts do not certify a real llama.cpp deployment or CUDA/ROCm hardware.
The context-budget diagnostic remains Ollama-only; the app and standalone
benchmark use the configured provider.
