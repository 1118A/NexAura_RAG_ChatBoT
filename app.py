import os
import base64
import streamlit as st
from src.rag_pipeline import process_pdf, ask_question
from config import PROVIDERS, Gemini_API_key, Groq_API_key, OpenAI_API_key

# Page config
st.set_page_config(
    page_title="NexAura RAG Chat",
    page_icon="🤖",
    layout="centered"
)

# Ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# Load CSS
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Base64 helper for local hero image
def get_image_as_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

# Initialize session state flags
if "current_file" not in st.session_state:
    st.session_state["current_file"] = None
if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "👋 **Hello!** I am your RAG Chatbot Assistant.\n\nTo begin, please use the sidebar to **upload and index a PDF document**. Once processed, I can retrieve relevant context and answer your questions directly!"
        }
    ]

# --- SIDEBAR CONFIGURATIONS ---
st.sidebar.markdown('<h1 class="claude-sidebar-title">NexAura Control Hub</h1>', unsafe_allow_html=True)

# Center image banner with float animation in sidebar
hero_path = os.path.join("assets", "rag_welcome_hero.png")
if os.path.exists(hero_path):
    img_b64 = get_image_as_base64(hero_path)
    if img_b64:
        st.sidebar.markdown(f"""
        <div class="hero-img-container">
            <img src="data:image/png;base64,{img_b64}" style="width: 100%; display: block;" alt="RAG Avatar" />
        </div>
        """, unsafe_allow_html=True)

st.sidebar.markdown("---")

# 1. Document Upload & Processing inside Sidebar
st.sidebar.markdown("### 📂 Upload & Process")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file:
    if st.session_state["current_file"] != uploaded_file.name:
        st.session_state["current_file"] = uploaded_file.name
        st.session_state["pdf_processed"] = False

    save_path = os.path.join("data", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Status indicators in sidebar
    if not st.session_state["pdf_processed"]:
        st.sidebar.warning("⚠️ PDF uploaded but not indexed.")
        if st.sidebar.button("⚙️ Process & Index PDF", use_container_width=True):
            loader_placeholder = st.sidebar.empty()
            with loader_placeholder:
                st.sidebar.markdown("""
                <div class="pulse-loader" style="padding:1rem;">
                    <div class="spinner-ring" style="width:28px; height:28px; margin-bottom:0.4rem;"></div>
                    <div class="loader-text" style="font-size:0.85rem;">Indexing PDF...</div>
                </div>
                """, unsafe_allow_html=True)
            
            try:
                process_pdf(save_path)
                st.session_state["pdf_processed"] = True
                loader_placeholder.empty()
                st.sidebar.success("🎉 Indexed successfully!")
                # Insert system message in chat
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"📑 **System Alert**: Document *{uploaded_file.name}* has been successfully processed and indexed! You can now start querying."
                })
                st.rerun()
            except Exception as e:
                loader_placeholder.empty()
                st.sidebar.error(f"Failed: {str(e)}")
    else:
        st.sidebar.success(f"✔️ Active: {uploaded_file.name}")

st.sidebar.markdown("---")

# 2. LLM Model Settings
st.sidebar.markdown("### 🤖 Model Configurations")
provider_list = list(PROVIDERS.keys())
default_provider_index = provider_list.index("Groq") if "Groq" in provider_list else 0

provider = st.sidebar.selectbox(
    "LLM Provider",
    options=provider_list,
    index=default_provider_index
)

available_models = PROVIDERS[provider]["models"]
model_name = st.sidebar.selectbox(
    "Model Option",
    options=available_models,
    index=0
)

# 3. Keys and Security
key_choice = st.sidebar.radio(
    "API Key Settings",
    options=["Use Default Key", "Provide My Own Key"],
    index=0
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
        st.sidebar.warning(f"⚠️ No default key for {provider}.")
else:
    api_key = st.sidebar.text_input(
        f"Enter {provider} API Key",
        type="password",
        placeholder="Paste credentials..."
    )

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "👋 Chat history cleared. What would you like to ask about the indexed document?"
        }
    ]
    st.rerun()

# --- MAIN UI AREA (CHAT INTERFACE) ---

st.markdown('<h1 class="claude-title">NexAura RAG Chat</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-desc">Retrieve and synthesize answers from documents using cutting edge LLMs</p>', unsafe_allow_html=True)

# Render Chat History
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Helper to show loader
def show_loader(message, subtext="Please wait..."):
    return st.markdown(f"""
    <div class="pulse-loader">
        <div class="spinner-ring"></div>
        <div class="loader-text">{message}</div>
        <div class="loader-subtext">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

# Chat Input at bottom
if prompt := st.chat_input("Ask a question about the document..."):
    # Render and append user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Error checking before retrieval
    if not st.session_state.get("pdf_processed"):
        warn_msg = "⚠️ **Cannot answer query**: Please upload a PDF file and click **Process & Index PDF** in the sidebar first."
        with st.chat_message("assistant"):
            st.markdown(warn_msg)
        st.session_state["messages"].append({"role": "assistant", "content": warn_msg})
    elif not api_key:
        err_msg = f"❌ **Credentials missing**: Please provide or configure an API key for {provider} in the sidebar."
        with st.chat_message("assistant"):
            st.markdown(err_msg)
        st.session_state["messages"].append({"role": "assistant", "content": err_msg})
    else:
        # Show loader widget while generating answer
        loader_placeholder = st.empty()
        with loader_placeholder:
            show_loader(
                "Retrieving chunks & generating answer...",
                f"Running {provider} model: {model_name}"
            )
        
        try:
            answer = ask_question(
                question=prompt,
                provider=provider,
                model_name=model_name,
                api_key=api_key
            )
            loader_placeholder.empty()
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        except Exception as e:
            loader_placeholder.empty()
            err_text = f"❌ **Generation Failed**:\n\n{str(e)}"
            with st.chat_message("assistant"):
                st.markdown(err_text)
            st.session_state["messages"].append({"role": "assistant", "content": err_text})
            
    st.rerun()