from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from inference import SteelClassifier


DEMO_DIR = Path(__file__).resolve().parent
WEIGHTS = DEMO_DIR.parent / "ResNet_best_acc.pth"
EXAMPLES = DEMO_DIR / "examples.png"
EMPTY_RESULT = """
<section class="prediction" aria-live="polite">
  <p class="result-label">预测结果</p>
  <h2>等待上传图片</h2>
  <p class="hint">上传一张钢材表面图片，查看缺陷类别和模型置信度。</p>
</section>
"""


def render_result(ranked):
    (_, english, chinese), _ = ranked[0]
    rows = []
    for (_, name, translation), score in ranked:
        percentage = score * 100
        rows.append(f"""
          <div class="score">
            <div class="score-label"><span>{name} · {translation}</span>
              <strong>{percentage:.1f}%</strong></div>
            <div class="score-track" aria-hidden="true">
              <div class="score-fill" style="width: {percentage:.3f}%"></div>
            </div>
          </div>""")
    return f"""
    <section class="prediction" aria-live="polite">
      <p class="result-label">预测结果</p>
      <h2>{english} · {chinese}</h2>
      {''.join(rows)}
      <p class="hint">百分比为模型对当前图片的 Softmax 置信度。</p>
    </section>
    """


def build_demo(classifier):
    def classify(filepath):
        if not filepath:
            return initial_image, EMPTY_RESULT
        try:
            with Image.open(filepath) as source:
                if source.format not in {"PNG", "JPEG", "BMP", "WEBP"}:
                    raise ValueError("Unsupported format")
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError):
            return initial_image, """
            <section class="prediction" role="alert">
              <p class="result-label">预测结果</p>
              <h2>无法读取图片</h2>
              <p class="hint">请上传有效的 JPG、PNG、BMP 或 WebP 图片。</p>
            </section>
            """
        ranked = classifier.predict(image)
        image.thumbnail((960, 960), Image.Resampling.LANCZOS)
        return image, render_result(ranked)

    initial_image = None
    if EXAMPLES.is_file():
        with Image.open(EXAMPLES) as montage:
            initial_image = montage.copy()
    with gr.Blocks(title="SteelDefect AI", analytics_enabled=False) as demo:
        with gr.Column(elem_id="steel-demo"):
            gr.HTML("""
              <header class="app-heading">
                <div><h1>SteelDefect AI</h1>
                  <p>钢材表面缺陷智能分类</p></div>
                <span>ResNet-18</span>
              </header>
            """, container=False)
            with gr.Row(elem_id="demo-columns"):
                with gr.Column(min_width=280):
                    preview = gr.Image(
                        value=initial_image,
                        label="钢材表面图片预览",
                        show_label=False,
                        interactive=False,
                        buttons=[],
                        format="png",
                        height=380,
                        elem_id="steel-preview",
                    )
                    upload = gr.UploadButton(
                        "上传钢材图片",
                        file_types=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
                        elem_id="steel-upload",
                    )
                    gr.HTML('<p class="hint upload-hint">JPG、PNG、BMP、WebP</p>',
                            container=False)
                with gr.Column(min_width=280):
                    result = gr.HTML(EMPTY_RESULT, container=False)
            upload.upload(classify, upload, [preview, result], api_name="predict")
    return demo


if __name__ == "__main__":
    demo = build_demo(SteelClassifier(WEIGHTS))
    demo.launch(
        css=(DEMO_DIR / "style.css").read_text(encoding="utf-8"),
        server_name="127.0.0.1",
        share=False,
    )
