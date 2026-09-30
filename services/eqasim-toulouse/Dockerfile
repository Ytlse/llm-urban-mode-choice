FROM ubuntu:24.04

LABEL org.opencontainers.image.description="""\
This image is used to run the eqasim pipeline \
"""

# Install system dependencies
RUN apt update && apt install -y git openjdk-25-jdk maven

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Setup environment
COPY pyproject.toml uv.lock .python-version /eqasim/
RUN cd /eqasim && uv sync

# Copy eqasim code (data is provided at runtime via volume mount at /eqasim-data)
COPY analysis /eqasim/analysis
COPY matsim /eqasim/matsim
COPY noise /eqasim/noise
COPY synthesis /eqasim/synthesis
COPY data /eqasim/data
COPY version.txt /eqasim

# Set up iterface
WORKDIR /eqasim
ENTRYPOINT uv run -m synpp /eqasim/config.yml --working-directory /eqasim-cache --data_path /eqasim-data --output_path /eqasim-output
