# Cleanup Summary

## Removed legacy UI and artifacts
- `ui/` (Tkinter/PySimpleGUI desktop UI) — replaced by React frontend and FastAPI backend.
- `templates/` and `static/` — Flask UI assets removed after migration.
- `web_app.py` — Flask-based web server removed in favor of FastAPI.
- `app.py` — legacy entrypoint removed after web migration.
- `attacks.py` (root) — moved into `na_core/attacks.py` to centralize core logic.
- `core/` — renamed to `na_core/` to match new architecture.

## Why
These files were tied to legacy UI or duplicated core logic. The project now uses:
- `na_core/` for dataset/model/attack logic,
- `backend/` for FastAPI services,
- `frontend/` for React UI.
