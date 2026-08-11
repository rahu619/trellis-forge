# trellis-forge — CUDA backend image.
# Target: Linux host + NVIDIA GPU >= 24 GB VRAM (A100/H100/4090-class).
#
# Packages the upstream microsoft/TRELLIS.2 install plus trellis-forge, so a
# GPU box needs only Docker + the NVIDIA container runtime. If upstream's
# setup.sh changes, adjust the TRELLIS.2 install block to match its README.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- Upstream TRELLIS.2 (code + build) ---------------------------------------
# Pinned clone: bump TRELLIS2_REF after verifying a newer commit works.
ARG TRELLIS2_REF=main
RUN git clone https://github.com/microsoft/TRELLIS.2 /opt/trellis2 \
    && cd /opt/trellis2 \
    && if [ "$TRELLIS2_REF" != "main" ]; then git checkout "$TRELLIS2_REF"; fi
WORKDIR /opt/trellis2
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && bash setup.sh

# --- trellis-forge -------------------------------------------------------------
COPY . /opt/trellis-forge
RUN pip install "/opt/trellis-forge[rembg,hf]"

WORKDIR /work
ENTRYPOINT ["trellis-forge"]
CMD ["--help"]
