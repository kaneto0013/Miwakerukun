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

## 運用ガイド

### 失敗時のロールバック手順

1. 運用中の API プロセスを停止します（`Ctrl+C` またはサービス管理ツールから停止）。
2. データベースとパラメータのバックアップを復元します。
   - `db/app.sqlite`
   - `db/parameters.json`（閾値 τ と重みのスナップショット）
3. コード変更が原因の場合は Git で該当コミットをロールバックします。
   ```bash
   git checkout HEAD~1  # 必要なコミットまで戻す
   ```
4. API を再起動して疎通を確認します。
   ```bash
   uvicorn src.api:app --reload
   ```
5. `/health` エンドポイントで状態を確認し、比較・フィードバック機能が期待通り動作することを手動確認します。

### しきい値調整の運用

1. まずドライランで直近フィードバックの傾向を確認します。
   ```bash
   python calibrate.py --window 200 --dry-run
   ```
   出力に含まれる現在値と校正後の精度を確認し、更新する価値があるか判断します。
2. 問題なければ実行して τ を更新し、`db/parameters.json` に反映します。
   ```bash
   python calibrate.py --window 200
   ```
3. API プロセスを再起動して新しい τ を読み込みます（常駐プロセスの場合）。
4. 任意の比較で総合スコアと判定が想定と合致するか確認し、必要に応じて追加のフィードバックで微調整します。
