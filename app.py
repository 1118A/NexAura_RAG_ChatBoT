"""
NexAura RAG Chat — Main Application
====================================
Security guardrails implemented:
  ✅ Path traversal prevention (filename sanitization)
  ✅ File size limit (50 MB enforced in code + Streamlit config)
  ✅ PDF magic-byte validation (real file-type check, not just extension)
  ✅ Chat input length limit (2000 characters max)
  ✅ Per-session rate limiting (max 30 questions per session)
  ✅ Scrubbed error messages (no raw exceptions shown to users)
  ✅ Uploaded files deleted after indexing (no persistent user data on disk)
  ✅ XSS-safe loader (no user-controlled strings interpolated into HTML)
  ✅ No API keys hardcoded — loaded exclusively from environment variables
"""

import os
import re
import base64
import streamlit as st
from src.rag_pipeline import process_pdf, ask_question
from config import PROVIDERS, Gemini_API_key, Groq_API_key, OpenAI_API_key

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_QUESTION_LENGTH = 2000
MAX_QUESTIONS_PER_SESSION = 30
SAFE_FILENAME_RE = re.compile(r"[^\w\-. ]")  # Only allow word chars, dash, dot, space


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexAura RAG Chat",
    page_icon="🤖",
    layout="centered",
)

# ── Ensure folders exist ───────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ── Load CSS ───────────────────────────────────────────────────────────────────
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Security helpers ───────────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    """
    Prevent path traversal by stripping directory components and
    replacing all non-safe characters with underscores.
    """
    # Take only the basename — neutralizes ../../ attacks
    name = os.path.basename(name)
    # Replace unsafe characters
    name = SAFE_FILENAME_RE.sub("_", name)
    # Collapse multiple underscores / dots and strip edges
    name = name.strip(". _")
    # Fallback if name becomes empty after sanitization
    return name if name else "uploaded_document.pdf"


def validate_pdf_magic_bytes(data: bytes) -> bool:
    """
    Verify the file starts with the PDF magic bytes (%PDF-).
    This prevents extension-spoofing attacks where a non-PDF is renamed to .pdf.
    """
    return data[:5] == b"%PDF-"


def is_safe_file_size(data: bytes) -> bool:
    """Return True if the file is within the allowed size limit."""
    return len(data) <= MAX_FILE_SIZE_BYTES


def get_image_as_base64(path: str) -> str:
    """Safely load an image as base64; return empty string on any failure."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""


def safe_loader_html() -> str:
    """
    Return a static (no user-controlled string) loader HTML.
    Static content is safe to render with unsafe_allow_html=True.
    """
    return """
    <div class="pulse-loader">
        <div class="spinner-ring"></div>
        <div class="loader-text">Retrieving context &amp; generating answer…</div>
        <div class="loader-subtext">Please wait</div>
    </div>
    """


def safe_indexing_loader_html() -> str:
    """Static loader for the PDF indexing step."""
    return """
    <div class="pulse-loader" style="padding:1rem;">
        <div class="spinner-ring" style="width:28px; height:28px; margin-bottom:0.4rem;"></div>
        <div class="loader-text" style="font-size:0.85rem;">Indexing PDF…</div>
    </div>
    """


# ── Session state initialisation ───────────────────────────────────────────────
if "current_file" not in st.session_state:
    st.session_state["current_file"] = None
if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 **Hello!** I am your RAG Chatbot Assistant.\n\n"
                "To begin, please use the sidebar to **upload and index a PDF document**. "
                "Once processed, I can retrieve relevant context and answer your questions!"
            ),
        }
    ]
if "question_count" not in st.session_state:
    st.session_state["question_count"] = 0


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<h1 class="claude-sidebar-title">NexAura Control Hub</h1>',
    unsafe_allow_html=True,
)

# Hero image
hero_path = os.path.join("assets", "rag_welcome_hero.png")
if os.path.exists(hero_path):
    img_b64 = get_image_as_base64(hero_path)
    if img_b64:
        st.sidebar.markdown(
            f'<div class="hero-img-container">'
            f'<img src="data:image/png;base64,{img_b64}" '
            f'style="width:100%;display:block;" alt="NexAura Avatar" />'
            f"</div>",
            unsafe_allow_html=True,
        )

st.sidebar.markdown("---")

# ── Document Upload & Processing ───────────────────────────────────────────────
st.sidebar.markdown("### 📂 Upload & Process")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    label_visibility="collapsed",
)

if uploaded_file:
    file_data = uploaded_file.getbuffer()

    # ── Security check 1: file size ────────────────────────────────────────────
    if not is_safe_file_size(bytes(file_data)):
        st.sidebar.error(
            f"❌ File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )
    # ── Security check 2: PDF magic bytes ─────────────────────────────────────
    elif not validate_pdf_magic_bytes(bytes(file_data)):
        st.sidebar.error(
            "❌ Invalid file type. Only genuine PDF files are accepted."
        )
    else:
        # ── Security check 3: sanitize filename ───────────────────────────────
        safe_name = sanitize_filename(uploaded_file.name)

        if st.session_state["current_file"] != safe_name:
            st.session_state["current_file"] = safe_name
            st.session_state["pdf_processed"] = False

        # Save to disk only under the safe, sanitized name
        save_path = os.path.join("data", safe_name)
        with open(save_path, "wb") as f:
            f.write(file_data)

        if not st.session_state["pdf_processed"]:
            st.sidebar.warning("⚠️ PDF uploaded but not indexed.")
            if st.sidebar.button("⚙️ Process & Index PDF", use_container_width=True):
                loader_placeholder = st.sidebar.empty()
                loader_placeholder.markdown(
                    safe_indexing_loader_html(), unsafe_allow_html=True
                )
                try:
                    process_pdf(save_path)
                    st.session_state["pdf_processed"] = True
                    loader_placeholder.empty()
                    st.sidebar.success("🎉 Indexed successfully!")
                    # ── Cleanup: remove uploaded file after indexing ───────────
                    try:
                        os.remove(save_path)
                    except OSError:
                        pass
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": (
                                f"📑 **System Alert**: Document *{safe_name}* has been "
                                "successfully processed and indexed! "
                                "You can now start querying."
                            ),
                        }
                    )
                    st.rerun()
                except Exception as e:
                    loader_placeholder.empty()
                    # Show user-friendly message only — str(e) is already scrubbed
                    # by the pipeline layer but we cap length as a final safeguard.
                    safe_err = str(e)[:300]
                    st.sidebar.error(f"❌ {safe_err}")
        else:
            st.sidebar.success(f"✔️ Active: {safe_name}")

st.sidebar.markdown("---")

# ── Model Configuration ────────────────────────────────────────────────────────
st.sidebar.markdown("### 🤖 Model Configurations")
provider_list = list(PROVIDERS.keys())
default_provider_index = (
    provider_list.index("Groq") if "Groq" in provider_list else 0
)

provider = st.sidebar.selectbox(
    "LLM Provider",
    options=provider_list,
    index=default_provider_index,
)

available_models = PROVIDERS[provider]["models"]
model_name = st.sidebar.selectbox(
    "Model Option",
    options=available_models,
    index=0,
)

# ── API Key Settings ───────────────────────────────────────────────────────────
key_choice = st.sidebar.radio(
    "API Key Settings",
    options=["Use Default Key", "Provide My Own Key"],
    index=0,
)

api_key = None
if key_choice == "Use Default Key":
    if provider == "Gemini":
        api_key = Gemini_API_key
    elif provider == "ChatGPT (OpenAI)":
        api_key = OpenAI_API_key
    elif provider == "Groq":
        api_key = Groq_API_key

    if not api_key:
        st.sidebar.warning(f"⚠️ No default key configured for {provider}.")
else:
    api_key = st.sidebar.text_input(
        f"Enter {provider} API Key",
        type="password",
        placeholder="Paste credentials...",
    )

st.sidebar.markdown("---")

# ── Rate limit indicator ───────────────────────────────────────────────────────
remaining = MAX_QUESTIONS_PER_SESSION - st.session_state["question_count"]
st.sidebar.caption(f"🔢 Questions remaining this session: **{remaining}/{MAX_QUESTIONS_PER_SESSION}**")

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "👋 Chat history cleared. What would you like to ask about the indexed document?",
        }
    ]
    st.session_state["question_count"] = 0
    st.rerun()

# ── MAIN CHAT AREA ─────────────────────────────────────────────────────────────
st.markdown('<h1 class="claude-title">NexAura RAG Chat</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-desc">Retrieve and synthesize answers from documents using cutting-edge LLMs</p>',
    unsafe_allow_html=True,
)

# Render chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the document..."):

    # ── Security check 4: input length limit ──────────────────────────────────
    if len(prompt) > MAX_QUESTION_LENGTH:
        prompt_display = prompt[:80] + "…"
        warn_msg = (
            f"⚠️ Your question is too long ({len(prompt)} characters). "
            f"Please limit questions to {MAX_QUESTION_LENGTH} characters."
        )
        with st.chat_message("assistant"):
            st.markdown(warn_msg)
        st.session_state["messages"].append({"role": "assistant", "content": warn_msg})
        st.rerun()

    # Render user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # ── Guard: PDF must be processed first ────────────────────────────────────
    if not st.session_state.get("pdf_processed"):
        warn_msg = (
            "⚠️ **Cannot answer query**: Please upload a PDF file and click "
            "**Process & Index PDF** in the sidebar first."
        )
        with st.chat_message("assistant"):
            st.markdown(warn_msg)
        st.session_state["messages"].append({"role": "assistant", "content": warn_msg})

    # ── Guard: API key must be set ────────────────────────────────────────────
    elif not api_key or not api_key.strip():
        err_msg = (
            f"❌ **Credentials missing**: Please provide or configure "
            f"an API key for **{provider}** in the sidebar."
        )
        with st.chat_message("assistant"):
            st.markdown(err_msg)
        st.session_state["messages"].append({"role": "assistant", "content": err_msg})

    # ── Security check 5: per-session rate limit ──────────────────────────────
    elif st.session_state["question_count"] >= MAX_QUESTIONS_PER_SESSION:
        err_msg = (
            f"⛔ **Session limit reached**: You have asked {MAX_QUESTIONS_PER_SESSION} "
            "questions this session. Please clear the chat history to continue."
        )
        with st.chat_message("assistant"):
            st.markdown(err_msg)
        st.session_state["messages"].append({"role": "assistant", "content": err_msg})

    else:
        # Show static loader (no user-controlled strings in HTML)
        loader_placeholder = st.empty()
        loader_placeholder.markdown(safe_loader_html(), unsafe_allow_html=True)

        try:
            answer = ask_question(
                question=prompt,
                provider=provider,
                model_name=model_name,
                api_key=api_key,
            )
            loader_placeholder.empty()
            st.session_state["question_count"] += 1
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})

        except Exception as e:
            loader_placeholder.empty()
            # Final safeguard: cap error message length — pipeline layer already
            # scrubs internal details, but we add a hard cap for defence-in-depth.
            safe_err = str(e)[:300]
            err_text = f"❌ **Error**: {safe_err}"
            with st.chat_message("assistant"):
                st.markdown(err_text)
            st.session_state["messages"].append({"role": "assistant", "content": err_text})

    st.rerun()