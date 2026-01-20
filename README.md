# NA_project — Web Adversarial Attacks Demo

Современное веб-приложение для демонстрации атак на STL-10 (96×96). Backend на FastAPI, frontend на React/Vite + Tailwind + shadcn/ui.

## Структура проекта

- `na_core/` — ядро: чтение STL-10 bin, загрузка модели, атаки/метрики.
- `backend/` — FastAPI API + job runner (runs/).
- `frontend/` — React/Vite + TypeScript + Tailwind UI.
- `runs/` — артефакты атак и экспорт результатов.
- `tests/` — контрактные тесты для инвариантов.
- `scripts/` — запуск dev окружения.

## Требования

- Python 3.11
- `torch`, `torchvision`, `pillow`, `fastapi`, `uvicorn`
- Node.js 18+ (для фронтенда)

## Быстрый запуск

**Linux/macOS:**
```
./scripts/dev.sh
```

**Windows (PowerShell):**
```
./scripts/dev.ps1
```
Logs are written to `logs/*.log`.

После запуска:
- Backend: http://localhost:8000
- Frontend: http://localhost:5173

## Основные сценарии

1) Выбор изображения по индексу в STL-10 (96×96) + превью.
2) Инференс: top-k, confidence, latency, true label.
3) Атаки (FGSM/BIM/PGD/DeepFool/C&W/AutoAttack) в фоне с прогрессом.
4) Таблица результатов атак + артефакты Original/Adv/Diff.

## API (минимум)

- `GET  /api/v1/health`
- `GET  /api/v1/attacks`
- `GET  /api/v1/defenses`
- `GET  /api/v1/dataset/info`
- `GET  /api/v1/images/{index}?format=png|raw`
- `POST /api/v1/infer`
- `POST /api/v1/jobs/attack`
- `GET  /api/v1/jobs/{job_id}`
- `WS   /api/v1/ws/jobs/{job_id}`
- `GET  /api/v1/jobs/{job_id}/artifacts/{attack_name}?type=original|adv|diff&format=png&amplify=...`
- `GET  /api/v1/jobs/{job_id}/export.csv`
- `GET  /api/v1/jobs/{job_id}/export.json`

## Инварианты

- **Bin reader:** использует исходную логику чтения STL-10 `.bin` (см. `na_core/dataset.py`).
- **Model loader:** загрузка модели/весов эквивалентна прежней реализации (см. `na_core/model_io.py`).
