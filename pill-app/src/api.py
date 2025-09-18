from __future__ import annotations

import html
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.params import Query
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from . import detect, feedback, schemas, store, visualize

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

app = FastAPI(title="pill-app")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/bags", response_model=schemas.Bag)
def create_bag(payload: Optional[schemas.BagCreate] = Body(default=None)) -> schemas.Bag:
    label = payload.label if payload else None
    bag = store.create_bag(label)
    return schemas.Bag(**bag)


@app.post("/api/images", response_model=schemas.Image)
async def upload_image(
    bag_id: str = Query(..., description="Identifier of the bag that owns the image"),
    file: UploadFile = File(...),
) -> schemas.Image:
    bag = store.get_bag_basic(bag_id)
    if bag is None:
        raise HTTPException(status_code=404, detail="Bag not found")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        pil_image = Image.open(BytesIO(contents))
        pil_image.load()
        width, height = pil_image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    suffix = Path(file.filename or "").suffix
    if not suffix:
        suffix = ".png"
    filename = f"{uuid.uuid4()}{suffix}"
    file_path = RAW_DIR / filename
    with open(file_path, "wb") as output:
        output.write(contents)

    relative_path = str(file_path.relative_to(BASE_DIR))
    record = store.create_image(bag_id=bag_id, path=relative_path, width=width, height=height)

    try:
        detections, image_size = detect.detect_pills(file_path)
    except Exception:
        detections = []
        image_size = (height, width)

    unrecognized_regions, _ = detect.identify_unrecognized_regions(detections, image_size)
    store.replace_detections(record["id"], detections)

    visualization_path = visualize.visualization_path_for(record["id"], OUTPUT_DIR)
    try:
        visualize.visualize_detections(file_path, detections, unrecognized_regions, visualization_path)
    except FileNotFoundError:
        pass

    return schemas.Image(**record)


@app.get("/api/bags/{bag_id}", response_model=schemas.BagWithImages)
def get_bag(bag_id: str) -> schemas.BagWithImages:
    bag = store.get_bag(bag_id)
    if bag is None:
        raise HTTPException(status_code=404, detail="Bag not found")
    return schemas.BagWithImages(**bag)


@app.get("/api/images/{image_id}", response_model=schemas.ImageDetections)
def get_image_with_detections(image_id: str) -> schemas.ImageDetections:
    image = store.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    detections = store.list_detections(image_id)
    image_path = BASE_DIR / image["path"]
    image_size = (int(image["height"]), int(image["width"]))

    unrecognized_regions, message = detect.identify_unrecognized_regions(detections, image_size)

    visualization_path = visualize.visualization_path_for(image_id, OUTPUT_DIR)
    if not visualization_path.exists() and image_path.exists():
        try:
            visualize.visualize_detections(image_path, detections, unrecognized_regions, visualization_path)
        except FileNotFoundError:
            pass

    relative_visualization_path = str(visualization_path.relative_to(BASE_DIR))

    return schemas.ImageDetections(
        image=schemas.Image(**image),
        detections=[schemas.Detection(**det) for det in detections],
        visualization_path=relative_visualization_path,
        unrecognized_regions=[schemas.RegionFlag(**region) for region in unrecognized_regions],
        message=message,
    )


@app.post("/api/compare", response_model=schemas.Comparison)
def create_comparison(payload: schemas.ComparisonCreate) -> schemas.Comparison:
    bag_a = store.get_bag_basic(payload.bag_id_a)
    bag_b = store.get_bag_basic(payload.bag_id_b)
    if bag_a is None or bag_b is None:
        raise HTTPException(status_code=404, detail="Bag not found")

    params = feedback.get_state()
    s_total = params.compute_total(
        [
            payload.score_color,
            payload.score_shape,
            payload.score_count,
            payload.score_size,
        ]
    )

    record = store.create_comparison(
        bag_id_a=payload.bag_id_a,
        bag_id_b=payload.bag_id_b,
        score_color=payload.score_color,
        score_shape=payload.score_shape,
        score_count=payload.score_count,
        score_size=payload.score_size,
        s_total=s_total,
        decision=payload.decision,
        preview_path=payload.preview_path,
    )

    logger.info(
        "Comparison %s stored (s_total=%.6f, w=%.4f, tau=%.4f)",
        record["id"],
        s_total,
        params.w,
        params.tau,
    )

    return schemas.Comparison(**record)


@app.post("/api/feedback", response_model=schemas.Feedback)
def submit_feedback(payload: schemas.FeedbackCreate) -> schemas.Feedback:
    comparison = store.get_comparison(payload.comparison_id)
    if comparison is None:
        raise HTTPException(status_code=404, detail="Comparison not found")

    feedback_record = store.create_feedback(
        comparison_id=payload.comparison_id,
        is_correct=payload.is_correct,
        note=payload.note,
    )

    params = feedback.get_state()
    params.register_feedback(bool(payload.is_correct))
    updated_total = params.compute_total(
        [
            comparison["score_color"],
            comparison["score_shape"],
            comparison["score_count"],
            comparison["score_size"],
        ]
    )
    store.update_comparison_total(payload.comparison_id, updated_total)

    logger.info(
        "Feedback %s applied to %s -> s_total %.6f",
        feedback_record["id"],
        payload.comparison_id,
        updated_total,
    )

    return schemas.Feedback(
        **feedback_record,
        updated_s_total=float(updated_total),
        parameters=schemas.ParameterSnapshot(w=params.w, tau=params.tau),
    )


@app.get("/compare/{comparison_id}", response_class=HTMLResponse)
def comparison_page(comparison_id: str) -> HTMLResponse:
    comparison = store.get_comparison(comparison_id)
    if comparison is None:
        raise HTTPException(status_code=404, detail="Comparison not found")

    params = feedback.get_state()
    feedback_entries = store.list_feedback(comparison_id)

    def _fmt(value: object) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "-"

    feedback_items = "".join(
        "<li>"
        + html.escape(entry["created_at"])
        + " : "
        + ("✔" if entry["is_correct"] else "✖")
        + (" - " + html.escape(entry["note"]) if entry.get("note") else "")
        + "</li>"
        for entry in feedback_entries
    )
    if not feedback_items:
        feedback_items = "<li>まだフィードバックはありません。</li>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang=\"ja\">
      <head>
        <meta charset=\"utf-8\" />
        <title>比較結果 {html.escape(comparison_id)}</title>
        <style>
          body {{ font-family: sans-serif; margin: 2rem; background-color: #f7f7f7; }}
          main {{ max-width: 720px; margin: 0 auto; background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }}
          h1 {{ margin-top: 0; font-size: 1.6rem; }}
          table {{ border-collapse: collapse; margin-bottom: 1rem; }}
          th, td {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid #ddd; text-align: left; }}
          .note-area {{ width: 100%; min-height: 5rem; margin-bottom: 0.8rem; }}
          .actions button {{ font-size: 1.2rem; padding: 0.6rem 1rem; margin-right: 0.4rem; cursor: pointer; border: none; border-radius: 6px; }}
          .actions button.correct {{ background-color: #2ecc71; color: white; }}
          .actions button.incorrect {{ background-color: #e74c3c; color: white; }}
          #status {{ margin-top: 1rem; min-height: 1.4rem; }}
          ul {{ padding-left: 1.2rem; }}
        </style>
      </head>
      <body>
        <main>
          <h1>比較結果</h1>
          <p><strong>ID:</strong> {html.escape(comparison_id)}</p>
          <p><strong>決定:</strong> {html.escape(str(comparison['decision']))}</p>
          <p><strong>現在の総合スコア:</strong> {_fmt(comparison['s_total'])}</p>
          <p><strong>パラメータ:</strong> w={params.w:.3f}, τ={params.tau:.3f}</p>
          <table>
            <tbody>
              <tr><th>色スコア</th><td>{_fmt(comparison['score_color'])}</td></tr>
              <tr><th>形状スコア</th><td>{_fmt(comparison['score_shape'])}</td></tr>
              <tr><th>個数スコア</th><td>{_fmt(comparison['score_count'])}</td></tr>
              <tr><th>サイズスコア</th><td>{_fmt(comparison['score_size'])}</td></tr>
            </tbody>
          </table>
          <section>
            <h2>フィードバック</h2>
            <textarea id=\"note\" class=\"note-area\" placeholder=\"メモを入力...\"></textarea>
            <div class=\"actions\">
              <button type=\"button\" class=\"correct\" data-feedback=\"1\">✔ 正しい</button>
              <button type=\"button\" class=\"incorrect\" data-feedback=\"0\">✖ 誤り</button>
            </div>
            <div id=\"status\"></div>
          </section>
          <section>
            <h3>これまでの履歴</h3>
            <ul>{feedback_items}</ul>
          </section>
        </main>
        <script>
          const sendFeedback = async (isCorrect) => {{
            const noteEl = document.getElementById('note');
            const statusEl = document.getElementById('status');
            statusEl.textContent = '送信中...';
            try {{
              const response = await fetch('/api/feedback', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                  comparison_id: '{html.escape(comparison_id)}',
                  is_correct: isCorrect,
                  note: noteEl.value || null
                }})
              }});
              const data = await response.json();
              if (!response.ok) {{
                statusEl.textContent = data.detail || '送信に失敗しました';
              }} else {{
                statusEl.textContent = `更新しました: s_total=${{data.updated_s_total.toFixed(3)}} (w=${{data.parameters.w.toFixed(3)}}, τ=${{data.parameters.tau.toFixed(3)}})`;
              }}
            }} catch (err) {{
              statusEl.textContent = '送信に失敗しました';
            }}
          }};

          document.querySelectorAll('[data-feedback]').forEach((button) => {{
            button.addEventListener('click', (event) => {{
              event.preventDefault();
              const value = parseInt(button.dataset.feedback, 10);
              sendFeedback(value);
            }});
          }});
        </script>
      </body>
    </html>
    """

    return HTMLResponse(content=html_content)
