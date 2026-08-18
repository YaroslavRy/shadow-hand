# ---------------------------------------------------------------------------
# Stage 1: build the browser MuJoCo-WASM bundle.
# Produces dist/ (index.html + JS bundle + ~10MB MuJoCo wasm + scene assets),
# which the runtime stage serves as the client-side execution path.
# ---------------------------------------------------------------------------
FROM node:20-slim AS web

WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY vite.config.js index.html scene_params.json ./
COPY web/ ./web/
COPY public/ ./public/
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: python runtime serving both paths.
#   /browser -> the static build from stage 1
#   /server  -> the Gradio app, rendering with MuJoCo via osmesa
# ---------------------------------------------------------------------------
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
COPY --from=web /build/dist ./dist

EXPOSE 7860
CMD ["python", "app.py"]
