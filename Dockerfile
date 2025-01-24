# cudnn 9
FROM nvcr.io/nvidia/pytorch:24.01-py3
# cudnn 8 
# FROM nvcr.io/nvidia/pytorch:21.12-py3
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHON_VERSION=3.10
ENV POETRY_VENV=/app/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION}-venv \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv into the correct location and update PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && export PATH="/root/.local/bin:$PATH" \
    && uv --version

RUN python -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==1.7.1

# Add Poetry venv and uv's bin directory to PATH
ENV PATH="/root/.local/bin:${POETRY_VENV}/bin:${PATH}"

WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.in-project true
RUN uv pip install -r <(poetry export --format=requirements.txt)

COPY . .
ENV CUDA_VISIBLE_DEVICES=0
RUN uv pip install wheel ninja packaging
RUN uv pip install flash-attn --no-build-isolation
RUN uv pip install python-multipart
RUN uv pip install faster-whisper
EXPOSE 9000

CMD gunicorn --bind 0.0.0.0:9000 --workers 1 --timeout 0 app.app:app -k uvicorn.workers.UvicornWorker