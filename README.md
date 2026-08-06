# Portrait Dataset Builder

Herramienta CLI para **construir datasets de retratos curados** a partir de
imágenes y video: detección y recorte de caras, deduplicación perceptual,
clasificación por CLIP y búsqueda vectorial.

Python 3.12, Typer + Rich + SQLAlchemy + OpenCV + InsightFace + CLIP.

---

## 🚀 Cómo correrlo

Requiere Python ≥ 3.12 y `uv`:

```bash
cd ~/Proyects/portrait-dataset-builder
uv sync --extra gpu   # o sin --extra gpu para CPU
uv run portrait-dataset --help
```

Entra al CLI interactivo de Typer. La API FastAPI también está disponible
(`portrait_dataset_builder.api`).

---

## 🧠 Qué hace

- **Ingesta**: imágenes locales, búsqueda web (DuckDuckGo) y video (yt-dlp).
- **Detección de caras**: InsightFace (recorta y enmarca retratos).
- **Deduplicación**: hashing perceptual (`imagehash`) y vectores (`usearch`).
- **Clasificación**: embeddings CLIP (`open-clip-torch`) para curar por
  contenido/semántica.
- **Almacenamiento**: metadatos en SQLite (async), pipeline orquestado con Hydra.

### Stack

Typer, Rich, loguru, SQLAlchemy (async), OpenCV, Pillow, numpy, insightface,
onnxruntime, imagehash, duckduckgo-search, yt-dlp, scikit-learn, polars,
usearch, open-clip-torch, hydra-core, FastAPI, transformers.

---

## 📁 Estructura

```
portrait-dataset-builder/
├── src/portrait_dataset_builder/
│   ├── cli/           ← entrypoint Typer (`portrait-dataset`)
│   ├── api/           ← rutas FastAPI
│   ├── pipeline/      ← pipeline de ingesta/procesamiento
│   ├── sources/       ← fuentes (local, web, video)
│   ├── compute/       ← detección, embeddings, hashing
│   ├── vector_backend.py ← búsqueda vectorial (usearch)
│   └── taxonomy.py    ← taxonomía/clasificación
├── configs/default.yaml  ← configuración Hydra
├── seeds/             ← seeds de datos
├── tests/             ← pytest (asyncio)
└── pyproject.toml     ← build hatchling, ruff, black, mypy estricto
```

---

## ⚙️ Notas

- Extras opcionales: `gpu` (torch/torchvision), `dev` (pytest, ruff, black, mypy).
- El `.venv` y los caches quedan fuera del repositorio.
- El script `who-question` y la app `who-app/` son de la fase de exploración de
  interfaz web y no forman parte del flujo principal.
