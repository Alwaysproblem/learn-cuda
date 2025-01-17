FROM alwaysproblem/fastdev-u2204:conda-nv12.2.0

RUN apt update -y && \
    apt install -y \
    cmake ninja-build \
    python3-pip \
    lsb-release wget software-properties-common gnupg && \
    apt clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" && \
    chmod +x llvm.sh && \
    ./llvm.sh 19 all && \
    rm -rf llvm.sh

RUN python3 -m pip install -U pip && \
    python3 -m pip install pre-commit
