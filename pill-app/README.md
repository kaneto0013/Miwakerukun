# pill-app

Minimal FastAPI project for managing "pill bags" and their images.

## Project structure

```
pill-app/
├── data/
│   ├── outputs/
│   └── raw/
├── db/
├── src/
│   ├── api.py
│   ├── schemas.py
│   └── store.py
├── tests/
├── README.md
└── requirements.txt
```

The SQLite database file (`db/app.sqlite`) is created automatically on first run. Uploaded images are saved under `data/raw/`.

## Setup

```bash
cd pill-app
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn src.api:app --reload
```

The service listens on `http://127.0.0.1:8000` by default.

## Example requests

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Create a bag:

```bash
curl -X POST "http://127.0.0.1:8000/api/bags" \
  -H "Content-Type: application/json" \
  -d '{"label": "morning"}'
```

Upload an image to the bag (replace `<BAG_ID>` with the response from the previous command):

```bash
curl -X POST "http://127.0.0.1:8000/api/images?bag_id=<BAG_ID>" \
  -F "file=@path/to/image.png"
```

List bag details with images:

```bash
curl http://127.0.0.1:8000/api/bags/<BAG_ID>
```

## Tests

Run the pytest suite from the project root:

```bash
pytest
```
