FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=osmesa \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install --no-install-recommends -y \
    libglib2.0-0 \
    libgl1 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-space.txt .
RUN pip install -r requirements-space.txt
COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
