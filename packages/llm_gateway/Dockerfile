# LLM gateway image (api, worker, flower services of the docker-compose).
# Build context = repository root: the image embeds the three packages, because the worker
# needs the category bundle (mobility_llm) and its domain (mobility_core).
#   docker build -f llm_gateway/Dockerfile .
FROM python:3.12-slim

WORKDIR /app

# pip cache layer: third-party dependencies are only reinstalled if
# requirements.txt changes.
COPY packages/llm_gateway/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The three packages, installed without re-resolving dependencies (already in the previous
# layer). The metadata (entry point llm_gateway.categories) come from here; in
# development, docker-compose mounts the sources on top under /app/<package> and puts
# /app first in sys.path via PYTHONPATH (WORKDIR alone does not do it for the celery/uvicorn
# console scripts: their sys.path[0] is /usr/local/bin, not the current directory).
COPY packages/mobility_core ./build/mobility_core
COPY packages/llm_gateway   ./build/llm_gateway
COPY packages/mobility_llm  ./build/mobility_llm
RUN pip install --no-cache-dir --no-deps ./build/mobility_core ./build/llm_gateway ./build/mobility_llm \
 && rm -rf ./build

# Unprivileged user: nothing in the image requires root.
RUN useradd --create-home --uid 1000 gateway
USER gateway

EXPOSE 8000
CMD ["uvicorn", "llm_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
