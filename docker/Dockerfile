FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies, Python 3.11, and git-lfs
# - git, wget, curl, ca-certificates: standard tools
# - libglib2.0-0, libsm6, libxrender1, libxext6: common GUI/multimedia libs (kept from original)
# - libssl-dev, libffi-dev: for building Python packages with C extensions
# - software-properties-common: for add-apt-repository (to add PPA for Python 3.11)
# - git-lfs: required by nanotron setup instructions
# - python3.11, python3.11-dev, python3.11-distutils: specified Python version for nanotron
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libssl-dev \
    libffi-dev \
    software-properties-common \
    git-lfs && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set python3.11 as the default python and python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 10

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Upgrade pip, install setuptools (good practice), and install uv
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools
RUN python3.11 -m pip install --no-cache-dir uv

# Clone nanotron repository
RUN git clone https://github.com/huggingface/nanotron.git /opt/nanotron

# Set working directory to nanotron for its specific installations
WORKDIR /opt/nanotron

# Install PyTorch using uv (for CUDA 12.4, matching the base image)
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cu124

# Install nanotron core dependencies using uv
RUN uv pip install --system --no-cache -e .

# Install nanotron example script dependencies and fused kernels using uv
RUN uv pip install --system --no-cache \
    datasets \
    transformers \
    "datatrove[io]" \
    numba \
    wandb \
    ninja \
    triton \
    "flash-attn>=2.5.0" --no-build-isolation

# Set a general working directory for the user's application context
WORKDIR /app

# Copy and install user's requirements.txt (if it exists)
# This allows users to add their own dependencies on top of nanotron
COPY requirements.txt .
RUN if [ -f requirements.txt ]; then \
      echo "Found requirements.txt, installing dependencies..." && \
      uv pip install --system --no-cache -vvv -r requirements.txt; \
    else \
      echo "requirements.txt not found, skipping."; \
    fi

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

ARG LDAP_USERNAME

# Create local user and group
RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
    groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID} && \
    useradd -m -U -s /bin/bash -G ${LDAP_GROUPNAME},sudo -u ${LDAP_UID} ${LDAP_USERNAME} && \
    echo -e "${LDAP_USERNAME}\n${LDAP_USERNAME}" | passwd ${LDAP_USERNAME}

RUN echo 'root:patheimathos' | chpasswd
RUN echo -e "\n${LDAP_USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Git configuration (optional, you can also use a repo saved in the NFS)
RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
    git config --global user.name ${GITHUB_NAME}
RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
    git config --global user.email ${GITHUB_EMAIL}
RUN git config --global pull.rebase false

# Switch to user context
USER ${LDAP_USERNAME}
WORKDIR /scratch

# Transfer scripts
COPY --chown=${LDAP_USERNAME} ./entrypoint.sh .
RUN chmod +x ./entrypoint.sh

COPY --chown=${LDAP_USERNAME} ./startup.sh .
RUN chmod +x ./startup.sh

ENTRYPOINT ["/bin/bash"]