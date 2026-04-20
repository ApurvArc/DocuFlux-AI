import atexit
import hashlib
import os
import re
from collections import defaultdict
from datetime import datetime

import gradio as gr
import pandas as pd

from core.answer import answer_question, classify_input
from core.config import AVAILABLE_PROVIDERS
from core.extractors import extract_file, extract_url, structure_as_markdown
from core.session_ingest import ingest_document
from core.ingest import ingest_all
from core.sync_manager import restore_from_bucket, backup_to_bucket, start_background_sync
from core.session_manager import (
    add_to_session_size,
    clear_all_sessions,
    create_session,
    destroy_session,
    get_session_db_path,
    get_session_size,
    is_file_processed,
    mark_file_processed,
)
from evaluation.eval import evaluate_all_answers, evaluate_all_retrieval

atexit.register(clear_all_sessions)

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_SESSION_BYTES = 50 * 1024 * 1024
MAX_VISIBLE_DOCS = 5

MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0

_URL_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)


def trim_docs_list(md_text: str) -> str:
    """Keep only the latest MAX_VISIBLE_DOCS entries in the session docs markdown."""
    entries = [e.strip() for e in md_text.split("\n\n") if e.strip()]
    if len(entries) > MAX_VISIBLE_DOCS:
        trimmed = entries[-MAX_VISIBLE_DOCS:]
        hidden = len(entries) - MAX_VISIBLE_DOCS
        return f"*...and {hidden} more file(s)*\n\n" + "\n\n".join(trimmed)
    return "\n\n".join(entries)


def extract_text_content(content) -> str:
    """Safely extract plain text from a Gradio message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def is_url(text: str) -> bool:
    """Return True if the entire stripped input is a single URL."""
    return bool(_URL_RE.match(text.strip()))


def format_context(context):
    if not context:
        return "### Retrieved Context\n\n*No relevant context found.*"

    total = len(context)
    parts = [f"### Retrieved Context ({total} chunk{'s' if total > 1 else ''})\n"]

    for i, doc in enumerate(context, 1):
        source = doc.metadata.get("source", "Unknown")
        doc_type = doc.metadata.get("doc_type", "")

        if doc_type == "web":
            try:
                from urllib.parse import urlparse

                parsed = urlparse(source)
                display_source = parsed.netloc or source
            except Exception:
                display_source = source
        elif "\\" in source or "/" in source:
            display_source = source.replace("\\", "/").split("/")[-1]
        else:
            display_source = source

        badge = " `WEB`" if doc_type == "web" else ""
        content = " ".join(doc.page_content.strip().split())
        content = content.lstrip("#*->= \t")
        if len(content) > 400:
            content = content[:400].rsplit(" ", 1)[0] + "..."

        source_label = f"[{display_source}]({source})" if doc_type == "web" else display_source
        parts.append(f"**[{i}] {source_label}**{badge}\n\n{content}\n\n---")

    return "\n\n".join(parts)


def format_size(size_bytes):
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def chat_with_mode(history, mode, session_id, def_hist, cust_hist, def_ctx, cust_ctx, current_docs_md, provider):
    last_message = extract_text_content(history[-1]["content"])
    prior = history[:-1]

    if "Custom" in mode:
        if is_url(last_message):
            url = last_message.strip()
            source_name = url.split("//")[-1].split("/")[0]
            text = extract_url(url)

            if text.startswith("Error:"):
                history.append({"role": "assistant", "content": f"Could not scrape that URL: {text}"})
                return history, "*URL scraping failed.*", def_hist, history, def_ctx, "*URL scraping failed.*", current_docs_md, gr.update()

            text_size = len(text.encode("utf-8"))
            current_size = get_session_size(session_id)
            if current_size + text_size > MAX_SESSION_BYTES:
                history.append({"role": "assistant", "content": "Session storage full. Reset to free space."})
                return history, gr.update(), def_hist, history, def_ctx, gr.update(), current_docs_md, gr.update()

            structured_md = structure_as_markdown(text, source_name)
            chunks_added = ingest_document(structured_md, source_name, session_id)
            add_to_session_size(session_id, text_size)

            reply = f"Scraped **{source_name}** and indexed {chunks_added} chunks. Ask me anything about it."
            new_docs_md = (
                f"**{source_name}** - web ({chunks_added} chunks)"
                if "No documents yet" in current_docs_md
                else current_docs_md + f"\n\n**{source_name}** - web ({chunks_added} chunks)"
            )
            history.append({"role": "assistant", "content": reply})
            size_md = f"**Size:** {format_size(get_session_size(session_id))} / 50 MB"
            return history, "*Web page ingested as context.*", def_hist, history, def_ctx, "*Web page ingested as context.*", trim_docs_list(new_docs_md), size_md

        classification = classify_input(last_message, provider)

        if classification == "context":
            source_name = f"User Input {datetime.now().strftime('%H:%M:%S')}"
            structured_md = structure_as_markdown(last_message, source_name)
            chunks_added = ingest_document(structured_md, source_name, session_id)

            if chunks_added > 0:
                add_to_session_size(session_id, len(last_message.encode("utf-8")))
                reply = f"Added to context ({chunks_added} chunks indexed). Ask me anything about it."
                new_docs_md = (
                    f"**{source_name}** ({chunks_added} chunks)"
                    if "No documents yet" in current_docs_md
                    else current_docs_md + f"\n\n**{source_name}** ({chunks_added} chunks)"
                )
            else:
                reply = "Could not extract any text from that input."
                new_docs_md = current_docs_md

            history.append({"role": "assistant", "content": reply})
            ctx_display = "*Text was ingested as context - no retrieval performed.*"
            size_md = f"**Size:** {format_size(get_session_size(session_id))} / 50 MB"
            return history, ctx_display, def_hist, history, def_ctx, ctx_display, trim_docs_list(new_docs_md), size_md

        answer, context = answer_question(
            last_message,
            prior,
            session_id=session_id,
            provider=provider,
            allow_web_fallback=True,
            is_custom_mode=True
        )

        new_docs_md = current_docs_md
        web_sources = set()
        for doc in context:
            if doc.metadata.get("doc_type") == "web":
                source = doc.metadata.get("source", "Web Search")
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(source).netloc or source
                except Exception:
                    domain = source
                web_sources.add(domain)

        for domain in web_sources:
            if domain not in new_docs_md:
                entry = f"**{domain}** — web search"
                if "No documents yet" in new_docs_md:
                    new_docs_md = entry
                else:
                    new_docs_md += f"\n\n{entry}"

        formatted_ctx = format_context(context)
        history.append({"role": "assistant", "content": answer})
        return history, formatted_ctx, def_hist, history, def_ctx, formatted_ctx, trim_docs_list(new_docs_md), gr.update()

    # Default mode: Strict Vector DB only. No session merging, no web fallback.
    answer, context = answer_question(
        last_message, 
        prior, 
        session_id=None, 
        provider=provider,
        allow_web_fallback=False,
        is_custom_mode=False
    )
    formatted_ctx = format_context(context)
    history.append({"role": "assistant", "content": answer})
    return history, formatted_ctx, history, cust_hist, formatted_ctx, cust_ctx, gr.update(), gr.update()


def put_message_in_chatbot(message, history):
    return "", history + [{"role": "user", "content": message}]


def toggle_mode(mode, def_hist, cust_hist, def_ctx, cust_ctx):
    is_custom = "Custom" in mode
    return (
        gr.update(visible=is_custom),
        cust_hist if is_custom else def_hist,
        cust_ctx if is_custom else def_ctx,
    )


def handle_upload(files, session_id, current_docs_md):
    if not files:
        return current_docs_md, f"**Size:** {format_size(get_session_size(session_id))} / 50 MB", gr.update()

    new_docs = []

    for file in files:
        file_path = file if isinstance(file, str) else getattr(file, "name", str(file))
        file_size = os.path.getsize(file_path)
        source_name = os.path.basename(file_path)

        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        dedup_key = hasher.hexdigest()

        if is_file_processed(session_id, dedup_key):
            continue

        if file_size > MAX_FILE_BYTES:
            new_docs.append(f"Warning: **{source_name}** ({format_size(file_size)}) exceeds 10 MB limit.")
            continue

        current_size = get_session_size(session_id)
        if current_size + file_size > MAX_SESSION_BYTES:
            new_docs.append(f"Warning: **{source_name}** - Session full ({format_size(current_size)}).")
            continue

        text = extract_file(file_path)

        if text.startswith("Error:"):
            new_docs.append(f"Error: **{source_name}** - {text}")
            continue

        mark_file_processed(session_id, dedup_key)

        structured_md = structure_as_markdown(text, source_name)
        chunks_added = ingest_document(structured_md, source_name, session_id)

        if chunks_added > 0:
            add_to_session_size(session_id, file_size)
            new_docs.append(f"Success: **{source_name}** ({chunks_added} chunks)")
        else:
            new_docs.append(f"Warning: **{source_name}** - No text extracted.")

    if not new_docs:
        return current_docs_md, f"**Size:** {format_size(get_session_size(session_id))} / 50 MB", gr.update(value=None)

    if "No documents yet" in current_docs_md:
        updated_md = "\n\n".join(new_docs)
    else:
        updated_md = current_docs_md + "\n\n" + "\n\n".join(new_docs)

    return trim_docs_list(updated_md), f"**Size:** {format_size(get_session_size(session_id))} / 50 MB", gr.update(value=None)


def reset_current_mode(mode, old_session_id, def_hist, cust_hist, def_ctx, cust_ctx, docs_list, size_bar):
    if "Custom" in mode:
        destroy_session(old_session_id)
        new_id = create_session()
        cust_hist = []
        cust_ctx = "*Retrieved context will appear here*"
        return new_id, [], cust_ctx, "*No documents yet*", "Size: 0.00 MB / 50 MB", def_hist, cust_hist, def_ctx, cust_ctx, gr.update(value=None)

    def_hist = []
    def_ctx = "*Retrieved context will appear here*"
    return old_session_id, [], def_ctx, docs_list, size_bar, def_hist, cust_hist, def_ctx, cust_ctx, gr.update()


def _get_color(value: float, metric_type: str) -> str:
    if metric_type == "mrr":
        return "green" if value >= MRR_GREEN else ("orange" if value >= MRR_AMBER else "red")
    if metric_type == "ndcg":
        return "green" if value >= NDCG_GREEN else ("orange" if value >= NDCG_AMBER else "red")
    if metric_type == "coverage":
        return "green" if value >= COVERAGE_GREEN else ("orange" if value >= COVERAGE_AMBER else "red")
    if metric_type in ["accuracy", "completeness", "relevance"]:
        return "green" if value >= ANSWER_GREEN else ("orange" if value >= ANSWER_AMBER else "red")
    return "black"


def _metric_html(label: str, value: float, metric_type: str, is_percentage: bool = False, score_format: bool = False) -> str:
    color = _get_color(value, metric_type)
    value_str = f"{value:.1f}%" if is_percentage else (f"{value:.2f}/5" if score_format else f"{value:.4f}")
    return f"""
    <div style="margin:10px 0;padding:15px;background:#f5f5f5;border-radius:8px;border-left:5px solid {color};">
        <div style="font-size:14px;color:#666;margin-bottom:5px;">{label}</div>
        <div style="font-size:28px;font-weight:bold;color:{color};">{value_str}</div>
    </div>"""


def run_retrieval_evaluation(progress=gr.Progress()):
    total_mrr = total_ndcg = total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0

    for test, result, prog_value in evaluate_all_retrieval():
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage
        category_mrr[test.category].append(result.mrr)
        progress(prog_value, desc=f"Evaluating test {count}...")

    if count == 0:
        return "<div style='padding:20px;color:#856404;background:#fff3cd;'>No retrieval tests found.</div>", pd.DataFrame()

    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_coverage = total_coverage / count

    final_html = f"""<div style="padding:0;">
        {_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
        {_metric_html("Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
        <div style="margin-top:20px;padding:10px;background:#d4edda;border-radius:5px;
                    text-align:center;border:1px solid #c3e6cb;">
            <span style="font-size:14px;color:#155724;font-weight:bold;">
                Evaluation complete: {count} tests
            </span>
        </div></div>"""

    df = pd.DataFrame([
        {"Category": cat, "Average MRR": sum(v) / len(v)}
        for cat, v in category_mrr.items()
    ])
    return final_html, df


def run_answer_evaluation(progress=gr.Progress()):
    from evaluation.eval import AI_EVAL_ENABLED, _get_judge_client

    ai_available = AI_EVAL_ENABLED and (_get_judge_client() is not None)

    if not ai_available:
        return """<div style="padding:20px;background:#fff3cd;border:1px solid #ffc107;
                     border-radius:8px;border-left:5px solid #ffc107;margin:10px 0;">
            <div style="font-size:18px;font-weight:bold;color:#856404;margin-bottom:8px;">
                AI Evaluation Unavailable
            </div>
            <div style="color:#856404;font-size:14px;line-height:1.6;">
                LM Studio is not running (or <code>AI_EVAL_ENABLED=false</code> in .env).<br/>
                Start LM Studio with a loaded model, then click Run Evaluation again.
            </div></div>""", pd.DataFrame()

    total_accuracy = total_completeness = total_relevance = 0.0
    category_accuracy = defaultdict(list)
    count = skipped = 0

    for test, result, prog_value in evaluate_all_answers():
        count += 1
        if result is None:
            skipped += 1
        else:
            total_accuracy += result.accuracy
            total_completeness += result.completeness
            total_relevance += result.relevance
            category_accuracy[test.category].append(result.accuracy)
        progress(prog_value, desc=f"Evaluating test {count}...")

    scored = count - skipped
    if scored == 0:
        return """<div style="padding:20px;background:#fff3cd;border:1px solid #ffc107;
                     border-radius:8px;border-left:5px solid #ffc107;">
            <b style="color:#856404;">No scores available</b><br/>
            <span style="color:#856404;">All evaluations were skipped. Check LM Studio is running.</span>
        </div>""", pd.DataFrame()

    skip_note = "" if skipped == 0 else f"""
        <div style="margin-top:10px;padding:8px 12px;background:#fff3cd;border-radius:5px;
                    font-size:13px;color:#856404;">
            {skipped} test(s) skipped (AI eval returned no score)
        </div>"""

    final_html = f"""<div style="padding:0;">
        {_metric_html("Accuracy", total_accuracy / scored, "accuracy", score_format=True)}
        {_metric_html("Completeness", total_completeness / scored, "completeness", score_format=True)}
        {_metric_html("Relevance", total_relevance / scored, "relevance", score_format=True)}
        {skip_note}
        <div style="margin-top:20px;padding:10px;background:#d4edda;border-radius:5px;
                    text-align:center;border:1px solid #c3e6cb;">
            <span style="font-size:14px;color:#155724;font-weight:bold;">
                Evaluation complete: {scored}/{count} tests scored
            </span>
        </div></div>"""

    df = pd.DataFrame([
        {"Category": cat, "Average Accuracy": sum(v) / len(v)}
        for cat, v in category_accuracy.items()
    ])
    return final_html, df


def main():
    print("Initializing Storage Bucket Sync...")
    restore_from_bucket()

    print("Synchronizing Default Vector DB with data/raw...")
    ingest_all()
    
    print("Backing up initial Vector DB to Storage Bucket...")
    backup_to_bucket()
    start_background_sync(interval_seconds=300)

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])
    gradio_major = int(gr.__version__.split(".")[0])

    custom_css = """
    /* ── Hide share button on chat message bubbles ── */
    button[title="Share"],
    button[aria-label="Share"],
    button[title="share"],
    button[aria-label="share"] {
        display: none !important;
    }

    /* ── Thin, sleek scrollbar across chatbot + context panel ── */
    .chatbot *,
    .prose *,
    .svelte-* {
        scrollbar-width: thin;
        scrollbar-color: rgba(120,120,120,0.35) transparent;
    }
    *::-webkit-scrollbar {
        width: 5px;
        height: 5px;
    }
    *::-webkit-scrollbar-track {
        background: transparent;
    }
    *::-webkit-scrollbar-thumb {
        background-color: rgba(120,120,120,0.35);
        border-radius: 10px;
    }
    *::-webkit-scrollbar-thumb:hover {
        background-color: rgba(120,120,120,0.6);
    }
    """

    blocks_kwargs = {"title": "DocuFlux AI", "css": custom_css}
    launch_kwargs = {"ssr_mode": False}
    if gradio_major >= 6:
        launch_kwargs["theme"] = theme
    else:
        blocks_kwargs["theme"] = theme

    with gr.Blocks(**blocks_kwargs) as ui:
        session_state = gr.State(create_session())
        model_provider = gr.State(AVAILABLE_PROVIDERS[0])
        def_hist = gr.State([])
        cust_hist = gr.State([])
        def_ctx = gr.State("*Retrieved context will appear here*")
        cust_ctx = gr.State("*Retrieved context will appear here*")

        with gr.Row(equal_height=True):
            gr.Markdown("# DocuFlux AI\nAsk me anything about the documents!")

        with gr.Tabs():
            with gr.Tab("Assistant", id=0):
                with gr.Row():
                    mode = gr.Radio(
                        ["Default (Built-in KB)", "Custom (Upload Files)"],
                        value="Default (Built-in KB)",
                        label="Search Mode",
                        info="Default uses built-in KB. Custom uses uploaded docs first, then web fallback if needed.",
                        scale=2,
                    )
                    llm_selector = gr.Dropdown(
                        choices=AVAILABLE_PROVIDERS,
                        value=AVAILABLE_PROVIDERS[0],
                        label="LLM Provider",
                        info="Only configured providers are shown.",
                        scale=1,
                    )

                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot_kwargs = {"label": "Conversation", "height": 550}
                        if gradio_major < 6:
                            chatbot_kwargs["type"] = "messages"
                            chatbot_kwargs["show_copy_button"] = True
                        chatbot = gr.Chatbot(**chatbot_kwargs)
                        message = gr.Textbox(
                            placeholder="Message AI Assistant...",
                            show_label=False,
                        )
                        clear_btn = gr.Button("Clear Current Chat", variant="secondary")

                    with gr.Column(scale=1):
                        with gr.Group(visible=False) as session_info:
                            file_upload = gr.File(
                                label="Upload Files (max 10 MB each)",
                                file_types=[".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".docx"],
                                file_count="multiple",
                            )
                            gr.Markdown("**Session Documents**")
                            session_docs_list = gr.Markdown("*No documents yet*")
                            session_size_bar = gr.Markdown("Size: 0.00 MB / 50 MB")
                            gr.Markdown("---")

                        context_markdown = gr.Markdown(
                            label="Retrieved Context",
                            value="*Retrieved context will appear here*",
                            container=True,
                            height=550,
                        )

                mode.change(
                    toggle_mode,
                    inputs=[mode, def_hist, cust_hist, def_ctx, cust_ctx],
                    outputs=[session_info, chatbot, context_markdown],
                )
                llm_selector.change(lambda x: x, inputs=[llm_selector], outputs=[model_provider])
                file_upload.upload(
                    handle_upload,
                    inputs=[file_upload, session_state, session_docs_list],
                    outputs=[session_docs_list, session_size_bar, file_upload],
                )
                message.submit(
                    put_message_in_chatbot,
                    [message, chatbot],
                    [message, chatbot],
                ).then(
                    chat_with_mode,
                    inputs=[chatbot, mode, session_state, def_hist, cust_hist, def_ctx, cust_ctx, session_docs_list, model_provider],
                    outputs=[chatbot, context_markdown, def_hist, cust_hist, def_ctx, cust_ctx, session_docs_list, session_size_bar],
                )
                clear_btn.click(
                    reset_current_mode,
                    inputs=[mode, session_state, def_hist, cust_hist, def_ctx, cust_ctx, session_docs_list, session_size_bar],
                    outputs=[session_state, chatbot, context_markdown, session_docs_list, session_size_bar, def_hist, cust_hist, def_ctx, cust_ctx, file_upload],
                )

            with gr.Tab("Evaluation", id=1):
                gr.Markdown("## Retrieval Evaluation")
                retrieval_button = gr.Button("Run Retrieval Evaluation", variant="primary", size="lg")
                with gr.Row():
                    with gr.Column(scale=1):
                        retrieval_metrics = gr.HTML(
                            "<div style='padding:20px;text-align:center;color:#999;'>"
                            "Click 'Run Retrieval Evaluation' to start</div>"
                        )
                    with gr.Column(scale=1):
                        retrieval_chart = gr.BarPlot(
                            x="Category",
                            y="Average MRR",
                            title="Average MRR by Category",
                            y_lim=[0, 1],
                            height=400,
                        )

                gr.Markdown("## Answer Evaluation")
                answer_button = gr.Button("Run Answer Evaluation", variant="primary", size="lg")
                with gr.Row():
                    with gr.Column(scale=1):
                        answer_metrics = gr.HTML(
                            "<div style='padding:20px;text-align:center;color:#999;'>"
                            "Click 'Run Answer Evaluation' to start</div>"
                        )
                    with gr.Column(scale=1):
                        answer_chart = gr.BarPlot(
                            x="Category",
                            y="Average Accuracy",
                            title="Average Accuracy by Category",
                            y_lim=[1, 5],
                            height=400,
                        )

                retrieval_button.click(
                    fn=run_retrieval_evaluation,
                    outputs=[retrieval_metrics, retrieval_chart],
                )
                answer_button.click(
                    fn=run_answer_evaluation,
                    outputs=[answer_metrics, answer_chart],
                )

    ui.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
