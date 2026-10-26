# Image for the EvoAgent API and remote workers (Milestone 5). One image, two commands:
#   evoagent serve --host 0.0.0.0     (API)
#   evoagent worker                   (worker)
# Build: docker build -t evoagent-app -f docker/app.Dockerfile .
FROM python:3.12-slim

# Workers start sandbox containers through the host's Docker daemon, so they need the CLI only
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[server]"
# Benchmark tasks that jobs may reference with benchmark_task
COPY benchmarks ./benchmarks

RUN useradd --create-home --uid 1000 evoagent
USER evoagent
ENV PYTHONUNBUFFERED=1 EVOAGENT_WORKER_BENCHMARK_ROOTS=/app/benchmarks
EXPOSE 8000
CMD ["evoagent", "serve", "--host", "0.0.0.0", "--port", "8000"]
