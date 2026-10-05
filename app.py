import os

from dotenv import load_dotenv

load_dotenv()

from ui import build_app  # noqa: E402

demo = build_app()

if __name__ == "__main__":
    # Cloud Run injects PORT; 7860 is the local default
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
