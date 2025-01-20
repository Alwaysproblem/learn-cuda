FROM alwaysproblem/fastdev-u2204:conda-nv12.2.0

RUN apt update -y && \
    apt install -y \
    cmake ninja-build \
    python3-pip \
    lsb-release wget software-properties-common gnupg && \
    apt clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 19 all && \
    rm -rf llvm.sh

RUN python3 -m pip install -U pip && \
    python3 -m pip install pre-commit

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 10 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 10 && \
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 10 && \
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-19 10 && \
    update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-19 10 && \
    update-alternatives --install /usr/bin/lldb lldb /usr/bin/lldb-19 10 && \
    update-alternatives --install /usr/bin/lldb-dap lldb-dap /usr/bin/lldb-dap-19 10 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 30
