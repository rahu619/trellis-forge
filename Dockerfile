# trellis-forge — CUDA backend image.
# Target: Linux host + NVIDIA GPU >= 24 GB VRAM (A100/H100/4090-class).
#
# Packages upstream's own install (microsoft/TRELLIS.2's setup.sh) plus
# trellis-forge, so a GPU box needs only Docker + the NVIDIA container runtime.
# CUDA 12.4 with torch 2.6.0/cu124 is the combination upstream documents and
# tests against, so it is pinned here rather than floated.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# jammy ships python3.10 and upstream's env uses it, but trellis-forge needs
# >=3.11, hence deadsnakes. sudo and libjpeg-dev are here because setup.sh
# --basic shells out to `sudo apt install -y libjpeg-dev`; the apt lists are
# deliberately left in place so that mid-build call can resolve.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates sudo build-essential \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv libjpeg-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- Upstream TRELLIS.2 (code + build) ---------------------------------------
# Pinned clone: bump TRELLIS2_REF after verifying a newer commit works.
ARG TRELLIS2_REF=main
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2 /opt/trellis2 \
    && cd /opt/trellis2 \
    && if [ "$TRELLIS2_REF" != "main" ]; then git checkout "$TRELLIS2_REF"; fi

WORKDIR /opt/trellis2

# setup.sh has to be sourced, not executed: its help path uses `return`, and
# with no flags it only prints usage and installs nothing. --new-env is left out
# because it would create a separate conda py3.10 env; these flags build into
# /opt/venv instead.
RUN pip install --upgrade pip \
    && pip install torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && . ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec \
        --cumesh --o-voxel --flexgemm

# --- trellis-forge -------------------------------------------------------------
# --basic installs pillow-simd above; this pulls real Pillow>=10 back over it.
COPY . /opt/trellis-forge
RUN pip install /opt/trellis-forge

WORKDIR /work
ENTRYPOINT ["trellis-forge"]
CMD ["--help"]
