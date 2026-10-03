from __future__ import annotations

from pathlib import Path

import gradio as gr

from generation import ArtifactService, ChatService, GroqClient, format_citation
from ingestion import SourceIngestor
from retrieval import LocalEmbedder, Retriever
from storage import StorageManager

EXT_TO_TYPE = {".pdf": "pdf", ".pptx": "pptx", ".txt": "txt"}


def build_app(storage: StorageManager | None = None) -> gr.Blocks:
    storage = storage or StorageManager()
    embedder = LocalEmbedder()
    retriever = Retriever(storage, embedder)
    llm = GroqClient()
    ingestor = SourceIngestor(storage, embedder)
    chat = ChatService(storage, retriever, llm)
    artifacts = ArtifactService(storage, llm)

    # ---------- helpers ----------
    def choices():
        return [(f"{n['name']} ({n['id'][:6]})", n["id"]) for n in storage.list_notebooks()]

    def sources_rows(nb_id):
        if not nb_id:
            return []
        return [
            [s["name"], s["type"], s["status"], s["metadata"].get("chunk_count", ""), s["metadata"].get("error", "")]
            for s in storage.get_notebook(nb_id)["sources"]
        ]

    def history_for(nb_id):
        if not nb_id:
            return []
        return [{"role": m["role"], "content": m["content"]} for m in storage.load_chats(nb_id)]

    def artifact_choices(nb_id):
        if not nb_id:
            return gr.update(choices=[], value=None)
        recs = artifacts.list(nb_id)
        return gr.update(choices=[(r["filename"], r["id"]) for r in recs], value=recs[-1]["id"] if recs else None)

    def refresh_all(nb_id):
        """Reload every notebook-scoped panel after a switch."""
        return sources_rows(nb_id), history_for(nb_id), "", artifact_choices(nb_id), "", None

    # ---------- notebook actions ----------
    def create_nb(name):
        name = (name or "").strip()
        if not name:
            return gr.update(), "Enter a notebook name.", ""
        nb = storage.create_notebook(name)
        return gr.update(choices=choices(), value=nb["id"]), f"Created notebook '{name}'.", ""

    def rename_nb(nb_id, name):
        name = (name or "").strip()
        if not nb_id or not name:
            return gr.update(), "Select a notebook and enter the new name."
        storage.rename_notebook(nb_id, name)
        return gr.update(choices=choices(), value=nb_id), f"Renamed to '{name}'."

    def delete_nb(nb_id):
        if not nb_id:
            return gr.update(), "Select a notebook first."
        storage.delete_notebook(nb_id)
        opts = choices()
        return gr.update(choices=opts, value=opts[0][1] if opts else None), "Notebook deleted."

    # ---------- sources ----------
    def add_files(nb_id, files):
        if not nb_id:
            return sources_rows(nb_id), "Select or create a notebook first."
        if not files:
            return sources_rows(nb_id), "Choose at least one file."
        msgs = []
        for f in files:
            path = Path(f if isinstance(f, str) else f.name)
            stype = EXT_TO_TYPE.get(path.suffix.lower())
            if not stype:
                msgs.append(f"✗ {path.name}: unsupported type (use PDF, PPTX, TXT).")
                continue
            try:
                res = ingestor.ingest(nb_id, stype, path.name, str(path))
                msgs.append(f"✓ {path.name}: {res['chunk_count']} chunks indexed.")
            except Exception as exc:
                msgs.append(f"✗ {path.name}: {exc}")
        return sources_rows(nb_id), "\n".join(msgs)

    def add_url(nb_id, url):
        url = (url or "").strip()
        if not nb_id:
            return sources_rows(nb_id), "Select or create a notebook first.", url
        if not url:
            return sources_rows(nb_id), "Enter a URL.", url
        try:
            res = ingestor.ingest(nb_id, "url", url, url)
            return sources_rows(nb_id), f"✓ {url}: {res['chunk_count']} chunks indexed.", ""
        except Exception as exc:
            return sources_rows(nb_id), f"✗ Could not ingest URL: {exc}", url

    # ---------- chat ----------
    def respond(nb_id, question, method, history):
        history = list(history or [])
        question = (question or "").strip()
        if not nb_id:
            gr.Warning("Select or create a notebook first.")
            return history, "", ""
        if not question:
            return history, "", ""
        try:
            out = chat.ask(nb_id, question, method=method)
        except Exception as exc:
            gr.Warning(str(exc))
            return history, question, ""
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": out["answer"]}]
        cites = "\n".join(f"- {format_citation(c)}\n  > {c['snippet']}…" for c in out["citations"])
        return history, "", f"**Citations** ({method}, {out['latency_s']:.2f}s)\n\n{cites}"

    # ---------- artifacts ----------
    def make_artifact(nb_id, kind):
        if not nb_id:
            return gr.update(), "", None, "Select or create a notebook first."
        try:
            rec = artifacts.generate(nb_id, kind)
        except Exception as exc:
            return gr.update(), "", None, f"✗ {exc}"
        return artifact_choices(nb_id), rec["content"], rec["path"], f"✓ {kind} saved as {rec['filename']}."

    def show_artifact(nb_id, art_id):
        if not nb_id or not art_id:
            return "", None
        for r in artifacts.list(nb_id):
            if r["id"] == art_id:
                return artifacts.read(r), r["path"]
        return "", None

    # ---------- layout ----------
    with gr.Blocks(title="NotebookLM Clone") as demo:
        gr.Markdown("# NotebookLM Clone")
        with gr.Row():
            nb_dd = gr.Dropdown(label="Notebook", choices=choices(), scale=3)
            nb_name = gr.Textbox(label="Name (for create / rename)", scale=2)
            create_btn = gr.Button("Create")
            rename_btn = gr.Button("Rename")
            delete_btn = gr.Button("Delete", variant="stop")
        nb_status = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Sources"):
                files = gr.File(label="Upload PDF / PPTX / TXT", file_count="multiple", file_types=[".pdf", ".pptx", ".txt"])
                upload_btn = gr.Button("Ingest files")
                with gr.Row():
                    url_box = gr.Textbox(label="Web URL", scale=4)
                    url_btn = gr.Button("Ingest URL", scale=1)
                src_status = gr.Textbox(label="Ingestion status", interactive=False, lines=3)
                src_table = gr.Dataframe(headers=["Name", "Type", "Status", "Chunks", "Error"], interactive=False)

            with gr.Tab("Chat"):
                method = gr.Radio(["rerank", "vector"], value="rerank", label="Retrieval method")
                chatbot = gr.Chatbot(label="Chat", height=380)
                q = gr.Textbox(label="Ask a question about your sources", placeholder="Press Enter to send")
                citations_md = gr.Markdown()

            with gr.Tab("Artifacts"):
                with gr.Row():
                    report_btn = gr.Button("Generate Report")
                    quiz_btn = gr.Button("Generate Quiz")
                art_status = gr.Markdown()
                art_dd = gr.Dropdown(label="Saved artifacts", choices=[])
                art_view = gr.Markdown()
                art_file = gr.File(label="Download (.md)", interactive=False)

        panels = [src_table, chatbot, citations_md, art_dd, art_view, art_file]
        nb_dd.change(refresh_all, nb_dd, panels)
        create_btn.click(create_nb, nb_name, [nb_dd, nb_status, nb_name])
        rename_btn.click(rename_nb, [nb_dd, nb_name], [nb_dd, nb_status])
        delete_btn.click(delete_nb, nb_dd, [nb_dd, nb_status])
        upload_btn.click(add_files, [nb_dd, files], [src_table, src_status])
        url_btn.click(add_url, [nb_dd, url_box], [src_table, src_status, url_box])
        q.submit(respond, [nb_dd, q, method, chatbot], [chatbot, q, citations_md])
        report_btn.click(make_artifact, [nb_dd, gr.State("report")], [art_dd, art_view, art_file, art_status])
        quiz_btn.click(make_artifact, [nb_dd, gr.State("quiz")], [art_dd, art_view, art_file, art_status])
        art_dd.input(show_artifact, [nb_dd, art_dd], [art_view, art_file])
    return demo
