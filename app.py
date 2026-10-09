import os

from dotenv import load_dotenv

load_dotenv()

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from ui import CSS, THEME, build_app  # noqa: E402

demo = build_app()

# Serve via uvicorn instead of demo.launch(): launch() runs a localhost self-check that can
# fail intermittently on Cloud Run cold starts and crash the container.
app = gr.mount_gradio_app(FastAPI(), demo, path="/", theme=THEME, css=CSS)

if __name__ == "__main__":
    # Cloud Run injects PORT; 7860 is the local default
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
