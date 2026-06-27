# MoulSot Runpod Worker

Runpod Serverless worker for `atlasia/moulsot.v0.3`.

## What this repo contains

- `handler.py` with a Runpod serverless entrypoint
- the `qwen_asr` package used by the model
- a `Dockerfile` for GPU deployment on Runpod

## Local structure

- `MODEL_ID` defaults to `atlasia/moulsot.v0.3`
- audio input can be a file path, Gradio-style payload, or base64 audio bytes
- temp audio files are deleted after each request

## Build

```bash
docker build -t moulsot-runpod-worker .
```

## Run

```bash
docker run --gpus all -e MODEL_ID=atlasia/moulsot.v0.3 moulsot-runpod-worker
```

## Notes for Runpod

- start with `Serverless`
- keep `workersMin = 0`
- use the smallest GPU that can load the model
- keep the endpoint private or authenticated
