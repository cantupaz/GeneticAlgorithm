FROM mcr.microsoft.com/devcontainers/python:3

RUN python -m pip install --upgrade pip \
    && python -m pip install 'flit>=3.8.0'

ENV FLIT_ROOT_INSTALL=1

COPY pyproject.toml .
RUN touch README.md \
    && mkdir -p src/python_package \
    && python -m flit install --only-deps --deps develop \
    && rm -r pyproject.toml README.md src


# Install uv
RUN pip3 install uv

WORKDIR /usr/src

# Setup code
RUN git clone --depth 1 https://github.com/cantupaz/GeneticAlgorithm

USER vscode
ENV VIRTUAL_ENV="/home/vscode/.local/ga-venv"
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /tmp

# Install Python dependencies from requirements
COPY requirements.txt ./
RUN uv pip install -r requirements.txt

WORKDIR /workspaces

# Set the default shell to bash instead of sh
ENV SHELL /bin/bash