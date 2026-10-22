FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# openssh-client is required for devops.vast.provision readiness probes and
# SSH into rented vast.ai boxes (agents previously failed with missing `ssh`).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# uv manages Python 3.13+ for this repo (see pyproject.toml).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Pre-install the interpreter uv resolves for `requires-python >= 3.13`
# (currently 3.14) so the cold `uv sync` in bootstrap_local.sh reuses a cached
# interpreter instead of downloading one on first agent boot. Mirrors the same
# optimization in the sibling rl-harness Dockerfile.
RUN uv python install 3.14
