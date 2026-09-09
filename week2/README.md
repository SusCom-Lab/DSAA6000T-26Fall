# Week 2: persistent Llama CPU/GPU services

Three Docker Compose services stay running: `cpu-backend`, `gpu-backend`, and
`router`. Each backend loads the complete Llama 3.1 8B model once in BF16 and
reuses it across HTTP requests. Tokenization remains on CPU. The checkpoint
is mounted read-only and never copied into the image.

Prerequisites: a Linux server with Bash, Python 3.10+, curl, Docker Engine
with CDI support enabled, and the Docker Compose v2 plugin. Your user must
have access to the Docker daemon. GPU inference also requires a compatible
NVIDIA driver, working `nvidia-smi`, and NVIDIA Container Toolkit with
UUID-named CDI devices. The helper uses `nvidia-ctk cdi list` (JSON/YAML
specifications), or reads JSON specs in `/etc/cdi` and `/var/run/cdi` when
the toolkit CLI is unavailable.
See [NVIDIA CDI setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
and [Docker CDI support](https://docs.docker.com/reference/cli/docker/container/run/#cdi-devices).

Supply your own local Llama checkpoint, including configuration, tokenizer,
and safetensors weights. The demo does not download the model.

Create the local configuration before running the demo:

```bash
cd week2
cp -n env.example .env
${EDITOR:-vi} .env
```

Set `MODEL_PATH` to the checkpoint directory on your server. The real `.env`
is ignored by Git; `env.example` is the version-controlled template.
Set `IMAGE_NAME` to your local image tag and `GPU_INDEX` to the host GPU index
shown by `nvidia-smi`. Quote paths containing spaces; use Bash-compatible
`KEY=value` assignments because the wrapper sources this file.

```bash
./run_demo.sh build     # build image only
./run_demo.sh up        # start three persistent containers
./run_demo.sh status    # see running containers
./run_demo.sh logs      # watch model loading; Ctrl-C exits logs only
```

Wait until the backends report ready, then send requests:

```bash
set -a; source .env; set +a
curl -s "http://${ROUTER_BIND_HOST}:${ROUTER_PORT}/health" | python3 -m json.tool
./run_demo.sh route
curl -s "http://${ROUTER_BIND_HOST}:${ROUTER_PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain CPU versus GPU inference.","max_new_tokens":32,"prefer":"cpu"}'
```

The router probes health, prefers GPU by default, and retries on CPU if GPU
is loading, unavailable, busy, or fails. Responses identify the selected
backend, fallback reasons, generated text, first-token latency, decode rate,
and memory. Requests are limited to 512 input tokens and 128 generated tokens.
Each backend handles one inference request at a time.

Demonstrate an actual service outage:

```bash
./run_demo.sh fallback  # stop GPU service
./run_demo.sh route     # CPU handles request; fallback reason appears
./run_demo.sh restore   # restart GPU service and reload its model
./run_demo.sh down      # stop and remove all demo containers
```

The `.env` fields are:

| Variable | Purpose |
| --- | --- |
| `COMPOSE_PROJECT_NAME` | Compose project/container-name prefix |
| `IMAGE_NAME` | Local Docker image name and tag |
| `MODEL_PATH` | Absolute host checkpoint directory mounted at `/model` |
| `ROUTER_BIND_HOST` | Host interface that publishes the router |
| `ROUTER_PORT` | Published host port for client requests |
| `SERVICE_PORT` | Private port shared by the three containers |
| `CPU_THREADS` | PyTorch CPU thread count |
| `GPU_INDEX` | Host GPU index to use, e.g. `0` or `1` |

On `up` and `restore`, the wrapper resolves `GPU_INDEX` to its live UUID
and selects the matching CDI device. Only that GPU is exposed; it appears
as `cuda:0` inside the container, regardless of its host index. Missing or
invalid indices and missing UUID CDI entries stop GPU startup with an error.
There is no fallback to all GPUs or stale device-node mappings.
`GPU_DEVICE` is now an internal value computed by the wrapper; remove any
old `GPU_DEVICE` assignment from your local configuration.

Use `./run_demo.sh up` rather than direct `docker compose up` so UUID
resolution runs. After changing `GPU_INDEX`, run `./run_demo.sh up` again
to recreate the GPU service and reload its model on the selected GPU.
Build, status, logs, shutdown, and requests do not perform GPU discovery.
Without a GPU, use `./run_demo.sh cpu-only`; the router reports GPU unavailable.

Only the router is published. Keep `ROUTER_BIND_HOST=127.0.0.1` unless remote
clients must reach the demo. Both model copies remain allocated until their
services stop. Allow approximately 18–20 GiB GPU memory and sufficient host RAM
for model loading and the CPU copy.

This is a teaching service: health and failure drive placement; performance
is reported per request. CPU memory is process peak RSS; GPU memory is PyTorch
peak allocation. Latency excludes model loading, tokenization, and HTTP
overhead. A timed-out request may still finish while the router retries.

`service.py` implements the persistent HTTP router and model backends.
`router.py` only resolves the host GPU index to a CDI device; the old
one-shot inference CLI has been removed. `test_router.py` covers GPU selection
with mocked hardware calls; `test_service.py` covers the HTTP router.
Image downloads require Docker Hub access, and the build installs Transformers
from PyPI. A registry connection reset blocks the image build.
