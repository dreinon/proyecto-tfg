"""Launch the local professional score-super-resolution demonstrator."""

from pathlib import Path

from score_super_resolution.web_app import APP_CSS, build_demo

PROJECT_ROOT = Path(__file__).resolve().parent
demo = build_demo(PROJECT_ROOT)

if __name__ == "__main__":
    demo.launch(css=APP_CSS)
