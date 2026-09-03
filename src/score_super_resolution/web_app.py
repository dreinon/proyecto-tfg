"""Gradio interface for the bounded professional score-enhancement demonstrator."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path

import gradio as gr
import numpy as np

from score_super_resolution.application import DemonstratorError, ProfessionalInferenceService

APP_TITLE = "Partitura · Laboratorio de ampliación"

APP_CSS = """
:root {
  --paper: #f5f1e8;
  --paper-raised: #fffdf8;
  --ink: #1a171b;
  --muted: #50565b;
  --signal: #d50066;
  --signal-dark: #9b004a;
  --rule: #d4ccbd;
  --interface-font: "Source Sans Pro", ui-sans-serif, system-ui, sans-serif;
  --display-font: Georgia, "Times New Roman", serif;
}

body,
.gradio-container {
  font-family: var(--interface-font) !important;
}

.gradio-container {
  color-scheme: light;
  --body-background-fill: var(--paper) !important;
  --body-text-color: var(--ink) !important;
  --block-background-fill: var(--paper-raised) !important;
  --block-label-background-fill: var(--paper-raised) !important;
  --block-label-text-color: var(--muted) !important;
  --block-title-text-color: var(--muted) !important;
  --input-background-fill: var(--paper-raised) !important;
  --input-border-color: var(--rule) !important;
  --border-color-primary: var(--rule) !important;
  --background-fill-secondary: #eee7da !important;
  --button-secondary-background-fill: var(--paper-raised) !important;
  --button-secondary-text-color: var(--ink) !important;
  background:
    linear-gradient(
      90deg, transparent 0 5.5rem, rgba(213,0,102,.11) 5.5rem 5.6rem, transparent 5.6rem
    ),
    repeating-linear-gradient(0deg, transparent 0 39px, rgba(25,24,22,.035) 39px 40px),
    var(--paper) !important;
  color: var(--ink) !important;
  min-height: 100vh;
}

#score-shell { max-width: 1240px; margin: 0 auto; padding: 32px 24px 64px; }
.institutional-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 32px;
  padding: 0 0 22px;
}
.institutional-logos {
  display: flex;
  align-items: center;
  gap: 22px;
}
.institutional-logos img {
  display: block;
  width: auto;
  object-fit: contain;
}
.upv-logo { height: 48px; }
.etsinf-logo { height: 52px; }
.institutional-divider {
  width: 1px;
  height: 42px;
  background: var(--rule);
}
.academic-context {
  color: var(--ink);
  font-size: .8rem;
  font-weight: 600;
  line-height: 1.45;
  text-align: right;
}
.academic-context span {
  display: block;
  color: var(--muted);
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
}
#masthead {
  border-top: 7px solid var(--ink);
  border-bottom: 1px solid #989084;
  padding: 28px 0 26px;
  margin-bottom: 24px;
}
#masthead-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 250px;
  align-items: end;
  gap: 40px;
}
#masthead h1 {
  max-width: 850px;
  margin: 8px 0 0;
  color: var(--ink) !important;
  font-family: var(--display-font);
  font-size: clamp(3rem, 6.6vw, 5.4rem);
  font-weight: 600;
  line-height: .98;
  letter-spacing: -.045em;
  text-wrap: balance;
}
#masthead .intro {
  max-width: 740px;
  margin: 18px 0 0;
  color: var(--muted) !important;
  font-size: 1.08rem;
  font-weight: 400;
  line-height: 1.55;
}
.eyebrow {
  color: var(--signal-dark) !important;
  font-size: .78rem;
  font-weight: 700;
  letter-spacing: .18em;
  text-transform: uppercase;
}
.model-stamp {
  border-left: 3px solid var(--signal);
  padding: 2px 0 2px 16px;
}
.model-stamp span,
.model-stamp strong { display: block; }
.model-stamp span {
  color: var(--muted);
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.model-stamp strong {
  margin-top: 5px;
  color: var(--ink);
  font-family: var(--display-font);
  font-size: 1.05rem;
  font-weight: 600;
}
.workflow-heading { margin: 0 0 12px; }
.workflow-heading span {
  display: block;
  color: var(--signal-dark);
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
}
.workflow-heading h2 {
  margin: 3px 0 0;
  color: var(--ink);
  font-family: var(--display-font);
  font-size: 1.4rem;
  font-weight: 600;
  line-height: 1.2;
}
.score-card {
  overflow: hidden;
  border: 1px solid var(--rule) !important;
  border-radius: 3px !important;
  background: rgba(255,253,248,.94) !important;
  box-shadow: 0 12px 34px rgba(48,41,31,.07) !important;
}
.score-card > div { border-color: var(--rule) !important; }
.score-card .upload-container .or { color: var(--muted) !important; }
.score-card label,
.score-card .label-wrap {
  color: var(--muted) !important;
  font-family: var(--interface-font) !important;
  font-weight: 600 !important;
}
.score-card [data-testid$="-radio-label"] {
  border-color: var(--rule) !important;
  background: var(--paper-raised) !important;
  color: var(--ink) !important;
}
.score-card [data-testid$="-radio-label"].selected {
  border-color: #b9aa96 !important;
  background: #eee7da !important;
}
.score-card [data-testid$="-radio-label"] span {
  color: var(--ink) !important;
  font-weight: 600 !important;
}
#run-button {
  min-height: 50px;
  border: 1px solid var(--signal-dark) !important;
  border-radius: 3px !important;
  background: var(--signal) !important;
  color: white !important;
  font-family: var(--interface-font) !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  box-shadow: 0 6px 16px rgba(155,0,74,.16);
  transition: background-color .18s ease, box-shadow .18s ease, transform .18s ease;
}
#run-button:hover {
  background: var(--signal-dark) !important;
  box-shadow: 0 9px 22px rgba(155,0,74,.22);
  transform: translateY(-1px);
}
#evidence-panel {
  min-height: 58px;
  border-left: 3px solid var(--signal);
  padding: 10px 16px;
  background: var(--paper-raised) !important;
  color: var(--ink) !important;
}
#evidence-panel p,
#evidence-panel li { color: var(--ink) !important; font-weight: 400 !important; }
#evidence-panel strong { font-weight: 700 !important; }
#evidence-panel code {
  border: 1px solid var(--rule);
  background: #eee7da !important;
  color: #51372f !important;
}
.warning-note {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 12px;
  border: 1px solid #dca1bd;
  border-radius: 3px;
  padding: 15px 16px;
  background: #fae3ee;
  color: #5d1739;
  font-size: .9rem;
  font-weight: 400;
  line-height: 1.5;
}
.warning-mark {
  font-family: var(--display-font);
  font-size: 1.35rem;
  font-weight: 600;
  line-height: 1;
}
.warning-note strong { display: block; margin-bottom: 2px; font-weight: 700; }
.scope-strip {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  margin-top: 26px;
  border: 1px solid var(--rule);
  background: var(--rule);
}
.scope-item {
  min-height: 112px;
  padding: 18px 20px;
  background: rgba(255,253,248,.96);
}
.scope-item span {
  display: block;
  margin-bottom: 6px;
  color: var(--signal-dark);
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .13em;
  text-transform: uppercase;
}
.scope-item p {
  margin: 0;
  color: var(--muted) !important;
  font-size: .9rem;
  font-weight: 400;
  line-height: 1.5;
}
.scope-item strong { color: var(--ink); font-weight: 600; }

@media (max-width: 760px) {
  .gradio-container {
    background:
      repeating-linear-gradient(0deg, transparent 0 39px, rgba(25,24,22,.035) 39px 40px),
      var(--paper) !important;
  }
  #score-shell { padding: 18px 14px 40px; }
  .institutional-bar { align-items: flex-start; flex-direction: column; gap: 18px; }
  .institutional-logos { gap: 14px; }
  .upv-logo { height: 36px; }
  .etsinf-logo { height: 40px; }
  .institutional-divider { height: 32px; }
  .academic-context { text-align: left; }
  #masthead { padding: 20px 0; }
  #masthead-grid { grid-template-columns: 1fr; gap: 20px; }
  #masthead h1 { font-size: clamp(2.55rem, 13vw, 4rem); letter-spacing: -.035em; }
  #masthead .intro { font-size: 1rem; }
  .model-stamp { max-width: 260px; }
  .scope-strip { grid-template-columns: 1fr; }
  .scope-item { min-height: 0; }
}
footer { display: none !important; }
"""


def _svg_data_uri(path: Path) -> str:
    """Embed one trusted institutional asset without requiring external hosting."""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


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
    upv_logo = _svg_data_uri(project_root / "assets/branding/upv.svg")
    etsinf_logo = _svg_data_uri(project_root / "assets/branding/etsinf.svg")
    with gr.Blocks(title=APP_TITLE, delete_cache=(3600, 3600)) as demo:
        with gr.Column(elem_id="score-shell"):
            gr.HTML(
                f"""
                <div class="institutional-bar">
                  <div class="institutional-logos">
                    <img class="upv-logo" src="{upv_logo}"
                      alt="Universitat Politècnica de València">
                    <span class="institutional-divider" aria-hidden="true"></span>
                    <img class="etsinf-logo" src="{etsinf_logo}"
                      alt="Escola Tècnica Superior d'Enginyeria Informàtica">
                  </div>
                  <div class="academic-context">
                    <span>Treball de Fi de Grau · 2025/2026</span>
                    Grau en Ciència de Dades
                  </div>
                </div>
                <header id="masthead">
                  <div id="masthead-grid">
                    <div>
                      <div class="eyebrow">Superresolución responsable · TFG</div>
                      <h1>Partitura ampliada.<br>Original preservado.</h1>
                      <p class="intro">Genera una copia de consulta con un modelo EDSR adaptado
                      a notación musical y contrasta el resultado con la imagen de entrada.</p>
                    </div>
                    <div class="model-stamp" aria-label="Características del modelo">
                      <span>Modelo validado</span>
                      <strong>EDSR · x2 / x4 · local</strong>
                    </div>
                  </div>
                </header>
                """
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes="score-card"):
                    gr.HTML(
                        """
                        <div class="workflow-heading">
                          <span>Paso 01</span>
                          <h2>Prepara la entrada</h2>
                        </div>
                        """
                    )
                    source = gr.Image(
                        label="Imagen de partitura",
                        image_mode="RGB",
                        sources=["upload", "clipboard"],
                        type="numpy",
                        format="png",
                        height=420,
                        buttons=["fullscreen"],
                    )
                    scale = gr.Radio(
                        choices=["x2 · ampliación moderada", "x4 · ampliación intensa"],
                        value="x2 · ampliación moderada",
                        label="Escala de ampliación",
                    )
                    run = gr.Button(
                        "Generar copia de consulta", variant="primary", elem_id="run-button"
                    )
                    gr.HTML(
                        """
                        <div class="warning-note">
                          <div class="warning-mark" aria-hidden="true">!</div>
                          <div><strong>No es restauración automática</strong>
                          El modelo puede engrosar trazos o no recuperar alteraciones, ornamentos,
                          dígitos y texto cuando la entrada ya ha perdido esa información.</div>
                        </div>
                        """
                    )
                with gr.Column(scale=7, elem_classes="score-card"):
                    gr.HTML(
                        """
                        <div class="workflow-heading">
                          <span>Paso 02</span>
                          <h2>Examina el derivado</h2>
                        </div>
                        """
                    )
                    comparison = gr.ImageSlider(
                        label="Comparación · original / derivado",
                        type="numpy",
                        format="png",
                        buttons=["download", "fullscreen"],
                        height=420,
                        max_height=420,
                    )
                    output = gr.Image(
                        label="Derivado descargable",
                        type="numpy",
                        format="png",
                        height=420,
                        interactive=False,
                        buttons=["download", "fullscreen"],
                    )
                    evidence = gr.Markdown(
                        "El registro del derivado aparecerá tras la inferencia.",
                        elem_id="evidence-panel",
                    )
            gr.HTML(
                """
                <section class="scope-strip" aria-label="Alcance del demostrador">
                  <div class="scope-item">
                    <span>Uso previsto</span>
                    <p><strong>Ampliación visual asistida y reversible</strong> para facilitar la
                    consulta de una partitura.</p>
                  </div>
                  <div class="scope-item">
                    <span>Límite</span>
                    <p>No sustituye una fuente histórica, una edición musical autoritativa ni un
                    máster digital.</p>
                  </div>
                  <div class="scope-item">
                    <span>Privacidad local</span>
                    <p>No se conserva un historial. Los archivos temporales del componente web se
                    eliminan periódicamente.</p>
                  </div>
                </section>
                """
            )
        run.click(handler, inputs=[source, scale], outputs=[comparison, output, evidence])
    return demo.queue(default_concurrency_limit=1, max_size=8)
