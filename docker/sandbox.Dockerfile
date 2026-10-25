# Image for EvoAgent's DockerSandbox: Python and pytest, nothing else.
# Build:  docker build -t evoagent-sandbox -f docker/sandbox.Dockerfile .
# Use:    EVOAGENT_SANDBOX_BACKEND=docker, or --sandbox docker for benchmarks/run_benchmarks.py
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest
WORKDIR /workspace
