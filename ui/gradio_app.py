from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from generation import ArtifactService, ChatService, GroqClient
from ingestion import SourceIngestor
from retrieval import LocalEmbedder, Retriever
from storage import StorageManager

from .theme import THEME_JS

EXT_TO_TYPE = {".pdf": "pdf", ".pptx": "pptx", ".txt": "txt"}
ART_PLACEHOLDER = "_Generated reports and quizzes appear here._"


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
        """Render the notebook's sources as cards (HTML)."""
        if not nb_id:
            return '<div class="empty">Select or create a notebook to add sources.</div>'
        srcs = storage.get_notebook(nb_id)["sources"]
        if not srcs:
            return '<div class="empty">No sources yet.<br>Upload a file or paste a URL.</div>'
        pill = {"ingested": ("ok", "Ready"), "failed": ("bad", "Failed")}
        cards = []
        for s in srcs:
            cls, label = pill.get(s["status"], ("wait", s["status"].title()))
            chunks = s["metadata"].get("chunk_count", "")
            sub = f"{chunks} chunks" if chunks != "" else ""
            if s["status"] == "failed":
                sub = html.escape(str(s["metadata"].get("error", "")))[:90]
            cards.append(
                f'<div class="src"><span class="badge {s["type"]}">{s["type"].upper()}</span>'
                f'<div class="meta"><div class="name" title="{html.escape(s["name"])}">{html.escape(s["name"])}</div>'
                f'<div class="sub">{sub}</div></div><span class="pill {cls}">{label}</span></div>'
            )
        return '<div class="src-list">' + "".join(cards) + "</div>"

    def citations_html(citations, method="", latency=None):
        if not citations:
            return ""
        head = '<div class="cites-head"><b>Sources used</b>'
        if method:
            head += f'<span class="tag">{html.escape(method)}</span>'
        if latency is not None:
            head += f"<span>{latency:.2f}s retrieval</span>"
        head += "</div>"
        items = []
        for c in citations:
            loc = f" · p./slide {c['page_or_slide']}" if c.get("page_or_slide") not in ("", None) else ""
            name = html.escape(str(c["source_name"]))
            if c.get("url"):
                name = f'<a href="{html.escape(c["url"])}" target="_blank" rel="noopener">{name}</a>'
            items.append(
                f'<div class="cite"><span class="n">{c["n"]}</span><div>'
                f'<div class="src-line">{name}{loc}</div>'
                f'<div class="snip">{html.escape(c["snippet"])}…</div></div></div>'
            )
        return f'<div class="cites">{head}{"".join(items)}</div>'

    def last_citations(nb_id):
        if not nb_id:
            return ""
        for m in reversed(storage.load_chats(nb_id)):
            if m["role"] == "assistant":
                return citations_html(m.get("citations", []), m.get("method", ""), m.get("latency_s"))
        return ""

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
        recs = artifacts.list(nb_id) if nb_id else []
        view = artifacts.read(recs[-1]) if recs else ART_PLACEHOLDER
        return sources_rows(nb_id), history_for(nb_id), last_citations(nb_id), artifact_choices(nb_id), view, None

    # ---------- notebook actions ----------
    def create_nb(name):
        name = (name or "").strip()
        if not name:
            return gr.update(), "Enter a notebook name.", ""
        nb = storage.create_notebook(name)
        gr.Info(f"Notebook '{name}' created")
        return gr.update(choices=choices(), value=nb["id"]), f"Created notebook '{name}'.", ""

    def rename_nb(nb_id, name):
        name = (name or "").strip()
        if not nb_id or not name:
            return gr.update(), "Select a notebook and enter the new name."
        storage.rename_notebook(nb_id, name)
        return gr.update(choices=choices(), value=nb_id), f"Renamed to '{name}'."

    def delete_nb(nb_id):
        if nb_id == "__cancel__":  # user dismissed the confirm dialog
            return gr.update(), ""
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
        return sources_rows(nb_id), "\n\n".join(msgs)

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
        return history, "", citations_html(out["citations"], method, out["latency_s"])

    # ---------- artifacts ----------
    def make_artifact(nb_id, kind):
        if not nb_id:
            return gr.update(), ART_PLACEHOLDER, None, "Select or create a notebook first."
        try:
            rec = artifacts.generate(nb_id, kind)
        except Exception as exc:
            return gr.update(), ART_PLACEHOLDER, None, f"✗ {exc}"
        return artifact_choices(nb_id), rec["content"], rec["path"], f"✓ {kind} saved as {rec['filename']}."

    def show_artifact(nb_id, art_id):
        if not nb_id or not art_id:
            return ART_PLACEHOLDER, None
        for r in artifacts.list(nb_id):
            if r["id"] == art_id:
                return artifacts.read(r), r["path"]
        return ART_PLACEHOLDER, None

    # ---------- layout ----------
    with gr.Blocks(title="NotebookLM Clone", fill_width=True) as demo:
        gr.HTML(
            """<div class="hero"><div class="brand"><div class="logo">N</div>
            <div><h1>NotebookLM Clone</h1><p>Chat with your sources. Grounded answers, with citations.</p></div></div>
            <button id="theme-btn" class="theme-btn" title="Switch theme: Light, Dark, Dusk"
            onclick="window.cycleNbTheme && window.cycleNbTheme()">☀️ Light</button></div>"""
        )
        with gr.Row(equal_height=False):
            # ----- left: notebook + sources -----
            with gr.Column(scale=3, min_width=300, elem_classes="panel"):
                gr.HTML('<div class="panel-title">Notebook</div>')
                nb_dd = gr.Dropdown(label="Notebook", show_label=False, choices=choices(), container=False,
                                    info=None, interactive=True, value=None)
                nb_name = gr.Textbox(show_label=False, container=False, placeholder="Name to create or rename…")
                with gr.Row():
                    create_btn = gr.Button("＋ Create", variant="primary", size="sm", min_width=70)
                    rename_btn = gr.Button("Rename", size="sm", min_width=70)
                    delete_btn = gr.Button("Delete", variant="stop", size="sm", min_width=70)
                nb_status = gr.Markdown(elem_classes="status")

                gr.HTML('<div class="panel-title">Sources</div>')
                files = gr.File(label="Drop PDF, PPTX or TXT files", file_count="multiple",
                                file_types=[".pdf", ".pptx", ".txt"], height=120)
                upload_btn = gr.Button("Add files", variant="primary", size="sm")
                with gr.Row():
                    url_box = gr.Textbox(show_label=False, container=False, placeholder="https://… web page", scale=4)
                    url_btn = gr.Button("Add URL", size="sm", scale=1, min_width=90)
                src_status = gr.Markdown(elem_classes="status")
                src_table = gr.HTML(sources_rows(None))

            # ----- center: chat -----
            with gr.Column(scale=6, min_width=380, elem_classes="panel"):
                with gr.Row():
                    gr.HTML('<div class="panel-title" style="padding-top:10px">Chat</div>')
                    method = gr.Radio(
                        [("Reranked · precise", "rerank"), ("Vector · fast", "vector")],
                        value="rerank", show_label=False, container=False, min_width=360, elem_classes="seg",
                    )
                chatbot = gr.Chatbot(
                    show_label=False, height=470, layout="bubble",
                    placeholder="### Ask anything about your sources\nAnswers are grounded in your documents and cite where they came from.",
                )
                with gr.Row(elem_classes="chips"):
                    chip1 = gr.Button("Summarize the key points", size="sm")
                    chip2 = gr.Button("What are the main topics?", size="sm")
                    chip3 = gr.Button("List important terms", size="sm")
                with gr.Row(elem_classes="ask"):
                    q = gr.Textbox(show_label=False, container=False, scale=6, lines=1, max_lines=4,
                                   placeholder="Ask a question and press Enter…")
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)
                citations_md = gr.HTML()

            # ----- right: studio -----
            with gr.Column(scale=3, min_width=300, elem_classes="panel"):
                gr.HTML('<div class="panel-title">Studio</div>')
                gr.HTML('<div class="studio-hint">Generate a document from everything in this notebook.</div>')
                report_btn = gr.Button("📄  Generate report", variant="primary", elem_classes="card-btn")
                quiz_btn = gr.Button("📝  Generate quiz + answer key", elem_classes="card-btn")
                art_status = gr.Markdown(elem_classes="status")
                art_dd = gr.Dropdown(label="Saved artifacts", choices=[])
                art_view = gr.Markdown(elem_classes="doc-view", value=ART_PLACEHOLDER)
                art_file = gr.DownloadButton("⬇  Download .md", value=None, size="sm")

        panels = [src_table, chatbot, citations_md, art_dd, art_view, art_file]

        def on_load():
            opts = choices()  # fresh list on every page load; open the first notebook if any
            return gr.update(choices=opts, value=opts[0][1] if opts else None)

        demo.load(on_load, None, nb_dd)
        demo.load(None, None, None, js=THEME_JS)
        nb_dd.change(refresh_all, nb_dd, panels)
        create_btn.click(create_nb, nb_name, [nb_dd, nb_status, nb_name])
        rename_btn.click(rename_nb, [nb_dd, nb_name], [nb_dd, nb_status])
        delete_btn.click(
            delete_nb, nb_dd, [nb_dd, nb_status],
            js="(nb) => confirm('Delete this notebook and all its sources, chats and artifacts?') ? nb : '__cancel__'",
        )
        upload_btn.click(add_files, [nb_dd, files], [src_table, src_status], show_progress="full")
        url_btn.click(add_url, [nb_dd, url_box], [src_table, src_status, url_box], show_progress="full")
        ask_inputs, ask_outputs = [nb_dd, q, method, chatbot], [chatbot, q, citations_md]
        q.submit(respond, ask_inputs, ask_outputs)
        send_btn.click(respond, ask_inputs, ask_outputs)
        for chip, text in ((chip1, "Summarize the key points of my sources."),
                           (chip2, "What are the main topics covered in my sources?"),
                           (chip3, "List the important terms and define them briefly.")):
            chip.click(lambda t=text: t, None, q).then(respond, ask_inputs, ask_outputs)
        report_btn.click(make_artifact, [nb_dd, gr.State("report")], [art_dd, art_view, art_file, art_status])
        quiz_btn.click(make_artifact, [nb_dd, gr.State("quiz")], [art_dd, art_view, art_file, art_status])
        art_dd.input(show_artifact, [nb_dd, art_dd], [art_view, art_file])
    return demo
