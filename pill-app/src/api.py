from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.params import Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import feedback, image_ops, ocr, schemas, signature, store
from .ui.views import router as ui_router

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

app = FastAPI(title="pill-app")
app.mount("/files", StaticFiles(directory=BASE_DIR), name="files")
app.include_router(ui_router)

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/bags", response_model=schemas.Bag)
def create_bag(payload: Optional[schemas.BagCreate] = Body(default=None)) -> schemas.Bag:
    label = payload.label if payload else None
    bag = store.create_bag(label)
    return schemas.Bag(**bag)


@app.get("/api/bags", response_model=list[schemas.Bag])
def list_bags() -> list[schemas.Bag]:
    bags = store.list_bags()
    return [schemas.Bag(**bag) for bag in bags]


@app.post("/api/images", response_model=schemas.Image)
async def upload_image(
    bag_id: str = Query(..., description="Identifier of the bag that owns the image"),
    file: UploadFile = File(...),
) -> schemas.Image:
    try:
        return await image_ops.process_upload(bag_id, file)
    except image_ops.BagNotFoundError:
        raise HTTPException(status_code=404, detail="Bag not found")
    except image_ops.EmptyUploadError:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    except image_ops.InvalidImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")


@app.get("/api/bags/{bag_id}", response_model=schemas.BagWithImages)
def get_bag(bag_id: str) -> schemas.BagWithImages:
    bag = store.get_bag(bag_id)
    if bag is None:
        raise HTTPException(status_code=404, detail="Bag not found")
    return schemas.BagWithImages(**bag)


@app.get("/api/images/{image_id}", response_model=schemas.ImageDetections)
def get_image_with_detections(image_id: str) -> schemas.ImageDetections:
    try:
        return image_ops.load_image_with_detections(image_id)
    except image_ops.ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")


@app.post("/api/images/{image_id}/reanalyze", response_model=schemas.ImageDetections)
def reanalyze_image(image_id: str) -> schemas.ImageDetections:
    try:
        return image_ops.reanalyze_image(image_id)
    except image_ops.ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    except image_ops.InvalidImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image file missing")


@app.post("/api/compare", response_model=schemas.Comparison)
def create_comparison(payload: schemas.ComparisonCreate) -> schemas.Comparison:
    bag_a = store.get_bag_basic(payload.bag_id_a)
    bag_b = store.get_bag_basic(payload.bag_id_b)
    if bag_a is None or bag_b is None:
        raise HTTPException(status_code=404, detail="Bag not found")

    params = feedback.get_state()
    samples_a = store.list_samples_for_bag(payload.bag_id_a)
    samples_b = store.list_samples_for_bag(payload.bag_id_b)
    score_color = float(payload.score_color)
    score_shape = float(payload.score_shape)
    score_count = float(payload.score_count)
    score_size = float(payload.score_size)
    score_embed = signature.boe_similarity(samples_a, samples_b)

    s_total = params.compute_total(
        sim_embed=score_embed,
        sim_color=score_color,
        sim_count=score_count,
        sim_size=score_size,
        sim_text=0.0,
    )

    score_ocr = 0.0
    if ocr.is_score_ambiguous(s_total):
        logger.info(
            "Total score %.4f within ambiguous band %s -> running OCR",
            s_total,
            ocr.AMBIGUOUS_RANGE,
        )
        score_ocr = ocr.score_samples(samples_a, samples_b, BASE_DIR)
        s_total = params.compute_total(
            sim_embed=score_embed,
            sim_color=score_color,
            sim_count=score_count,
            sim_size=score_size,
            sim_text=score_ocr,
        )

    record = store.create_comparison(
        bag_id_a=payload.bag_id_a,
        bag_id_b=payload.bag_id_b,
        score_color=score_color,
        score_shape=score_shape,
        score_count=score_count,
        score_size=score_size,
        score_embed=score_embed,
        score_ocr=score_ocr,
        s_total=s_total,
        decision=payload.decision,
        preview_path=payload.preview_path,
    )

    logger.info(
        "Comparison %s stored (s_total=%.6f, score_embed=%.4f, score_ocr=%.4f, weights=%s, tau=%.4f)",
        record["id"],
        s_total,
        score_embed,
        score_ocr,
        ",".join(f"{w:.2f}" for w in params.weights),
        params.tau,
    )

    return schemas.Comparison(**record)


@app.post("/api/feedback", response_model=schemas.Feedback)
def submit_feedback(payload: schemas.FeedbackCreate) -> schemas.Feedback:
    comparison = store.get_comparison(payload.comparison_id)
    if comparison is None:
        raise HTTPException(status_code=404, detail="Comparison not found")

    operator = payload.operator.strip()
    if not operator:
        raise HTTPException(status_code=422, detail="Operator name must not be blank")

    feedback_record = store.create_feedback(
        comparison_id=payload.comparison_id,
        is_correct=payload.is_correct,
        operator=operator,
        note=payload.note,
    )

    params = feedback.get_state()
    params.register_feedback(bool(payload.is_correct))
    score_embed = float(comparison.get("score_embed") or 0.0)
    updated_total = params.compute_total(
        sim_embed=score_embed,
        sim_color=float(comparison["score_color"]),
        sim_count=float(comparison["score_count"]),
        sim_size=float(comparison["score_size"]),
        sim_text=0.0,
    )
    store.update_comparison_total(payload.comparison_id, updated_total)

    logger.info(
        "Feedback %s by %s applied to %s -> decision=%s, is_correct=%s, s_total %.6f (score_embed=%.4f, weights=%s)",
        feedback_record["id"],
        feedback_record["operator"],
        payload.comparison_id,
        comparison.get("decision"),
        "correct" if payload.is_correct else "incorrect",
        updated_total,
        score_embed,
        ",".join(f"{w:.2f}" for w in params.weights),
    )

    return schemas.Feedback(
        **feedback_record,
        updated_s_total=float(updated_total),
        parameters=schemas.ParameterSnapshot(weights=params.weights, tau=params.tau, w=params.w),
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
        + html.escape(entry.get("operator") or "-")
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
          <p><strong>パラメータ:</strong> weights={[f"{w:.3f}" for w in params.weights]}, τ={params.tau:.3f}</p>
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
            <div style=\"display:flex; gap:0.75rem; margin-bottom:0.5rem;\">
              <input id=\"operator\" type=\"text\" placeholder=\"担当者\" style=\"flex:0 0 200px; padding:0.4rem;\" />
              <textarea id=\"note\" class=\"note-area\" placeholder=\"メモを入力...\"></textarea>
            </div>
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
          const operatorEl = document.getElementById('operator');
          const storedOperator = window.localStorage && window.localStorage.getItem('pill-app-operator');
          if (storedOperator) {{
            operatorEl.value = storedOperator;
          }}

          const sendFeedback = async (isCorrect) => {{
            const noteEl = document.getElementById('note');
            const statusEl = document.getElementById('status');
            statusEl.textContent = '送信中...';
            const operator = operatorEl.value.trim();
            if (!operator) {{
              statusEl.textContent = '担当者名を入力してください';
              operatorEl.focus();
              return;
            }}
            try {{
              const response = await fetch('/api/feedback', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                  comparison_id: '{html.escape(comparison_id)}',
                  is_correct: isCorrect,
                  operator,
                  note: noteEl.value || null
                }})
              }});
              const data = await response.json();
              if (!response.ok) {{
                statusEl.textContent = data.detail || '送信に失敗しました';
              }} else {{
                const weights = data.parameters.weights || [];
                const primary = weights.length ? weights[0].toFixed(3) : '0.000';
                statusEl.textContent = `更新しました: s_total=${{data.updated_s_total.toFixed(3)}} (w₀=${primary}, τ=${{data.parameters.tau.toFixed(3)}})`;
                try {{
                  window.localStorage && window.localStorage.setItem('pill-app-operator', operator);
                }} catch (error) {{
                  console.warn('operator persistence failed', error);
                }}
                noteEl.value = '';
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
