"""Gradio interface for the bounded professional score-enhancement demonstrator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gradio as gr
import numpy as np

from score_super_resolution.application import DemonstratorError, ProfessionalInferenceService

APP_TITLE = "Partitura · Laboratorio de ampliación"

APP_CSS = """
:root {
  --paper: #f4efe3;
  --paper-deep: #e8ddc9;
  --ink: #1d1b18;
  --muted: #6f685e;
  --signal: #b7472a;
  --rule: rgba(29, 27, 24, 0.15);
}
.gradio-container {
  background:
    linear-gradient(90deg, transparent 0 7%, rgba(183,71,42,.12) 7% 7.12%, transparent 7.12%),
    repeating-linear-gradient(0deg, transparent 0 31px, rgba(29,27,24,.055) 31px 32px),
    var(--paper) !important;
  color: var(--ink) !important;
}
#score-shell { max-width: 1180px; margin: 0 auto; padding: 22px 8px 56px; }
#masthead {
  border-top: 8px solid var(--ink);
  border-bottom: 1px solid var(--ink);
  padding: 24px 0 18px;
  margin-bottom: 28px;
}
#masthead h1 {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: clamp(2.5rem, 6vw, 5.8rem);
  line-height: .88;
  letter-spacing: -.055em;
  margin: 0;
  max-width: 900px;
}
#masthead p { color: var(--muted); max-width: 760px; font-size: 1.05rem; margin-top: 18px; }
.eyebrow {
  color: var(--signal); font-weight: 800; letter-spacing: .16em; text-transform: uppercase;
}
.score-card {
  border: 1px solid var(--rule) !important; background: rgba(255,252,246,.78) !important;
}
#run-button { background: var(--signal) !important; border: 0 !important; color: white !important; }
#run-button:hover { filter: brightness(.9); transform: translateY(-1px); }
#evidence-panel { border-left: 4px solid var(--signal); padding-left: 16px; }
.warning-note {
  color: #5e291c; background: #f2d5c8; border: 1px solid #d69b86; padding: 14px 16px;
}
footer { display: none !important; }
"""


def _format_evidence(result: object) -> str:
    return (
        "### Derivado generado\n"
        f"- **Método:** `{result.method_id}`\n"
        f"- **Escala:** x{result.scale}\n"
        f"- **Tiempo de inferencia:** {result.elapsed_seconds:.2f} s\n"
        f"- **Dispositivo:** `{result.device}`\n"
        f"- **Checkpoint:** `{result.checkpoint_sha256[:16]}…`\n"
        f"- **Salida:** `{result.output_sha256[:16]}…`\n\n"
        "El resultado es un derivado de consulta. Conserva siempre el original."
    )


def create_handler(
    service: ProfessionalInferenceService,
) -> Callable[[np.ndarray | None, str], tuple[tuple[np.ndarray, np.ndarray], np.ndarray, str]]:
    """Return the UI callback separately so its behavior can be tested without a server."""

    def enhance_page(
        image: np.ndarray | None,
        scale_label: str,
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, str]:
        if image is None:
            raise gr.Error("Selecciona primero una imagen de partitura.")
        scales = {"x2 · ampliación moderada": 2, "x4 · ampliación intensa": 4}
        if scale_label not in scales:
            raise gr.Error("Selecciona una escala x2 o x4 válida.")
        scale = scales[scale_label]
        try:
            result = service.enhance(image, scale=scale)
        except DemonstratorError as error:
            raise gr.Error(str(error)) from error
        return (image, result.pixels), result.pixels, _format_evidence(result)

    return enhance_page


def build_demo(
    project_root: Path,
    *,
    service: ProfessionalInferenceService | None = None,
) -> gr.Blocks:
    """Build the image-only demonstrator; launching and deployment remain explicit actions."""

    inference = service or ProfessionalInferenceService(project_root)
    handler = create_handler(inference)
    with gr.Blocks(title=APP_TITLE, delete_cache=(3600, 3600)) as demo:
        with gr.Column(elem_id="score-shell"):
            gr.HTML(
                """
                <header id="masthead">
                  <div class="eyebrow">Superresolución responsable · TFG</div>
                  <h1>Partitura<br>ampliada, no inventada.</h1>
                  <p>Genera una copia de consulta con el EDSR adaptado a notación musical.
                  Compara siempre el resultado con la fuente antes de utilizarlo.</p>
                </header>
                """
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes="score-card"):
                    source = gr.Image(
                        label="01 · Partitura de entrada",
                        image_mode="RGB",
                        sources=["upload", "clipboard"],
                        type="numpy",
                        format="png",
                        buttons=["fullscreen"],
                    )
                    scale = gr.Radio(
                        choices=["x2 · ampliación moderada", "x4 · ampliación intensa"],
                        value="x2 · ampliación moderada",
                        label="02 · Escala",
                    )
                    run = gr.Button(
                        "Generar copia de consulta", variant="primary", elem_id="run-button"
                    )
                    gr.HTML(
                        """
                        <div class="warning-note"><strong>No es restauración automática.</strong>
                        El modelo puede engrosar trazos o no recuperar alteraciones, ornamentos,
                        dígitos y texto cuando la entrada ha perdido esa información.</div>
                        """
                    )
                with gr.Column(scale=7, elem_classes="score-card"):
                    comparison = gr.ImageSlider(
                        label="03 · Original / derivado",
                        type="numpy",
                        format="png",
                        buttons=["download", "fullscreen"],
                        max_height=650,
                    )
                    output = gr.Image(
                        label="04 · Derivado descargable",
                        type="numpy",
                        format="png",
                        interactive=False,
                        buttons=["download", "fullscreen"],
                    )
                    evidence = gr.Markdown(
                        "El registro del derivado aparecerá tras la inferencia.",
                        elem_id="evidence-panel",
                    )
            gr.Markdown(
                "**Uso previsto:** ampliación visual asistida y reversible. "
                "**No usar como:** fuente histórica, edición musical autoritativa, entrada OMR "
                "validada o sustituto del máster digital. La aplicación no conserva un historial; "
                "los temporales del componente web se eliminan periódicamente."
            )
        run.click(handler, inputs=[source, scale], outputs=[comparison, output, evidence])
    return demo.queue(default_concurrency_limit=1, max_size=8)
