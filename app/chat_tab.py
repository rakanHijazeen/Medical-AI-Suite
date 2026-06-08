import streamlit as st
import io
from pypdf import PdfReader
from app.rag.chat_engine import MedicalChatEngine

# Initialize the backend orchestrator
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = MedicalChatEngine()

class PDFTextExtractor:
    @staticmethod
    def extract_text_from_stream(file_stream: io.BytesIO) -> str:
        """Extracts raw text pages from an in-memory PDF file stream."""
        try:
            reader = PdfReader(file_stream)
            extracted_text_chunks = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_text_chunks.append(f"--- PAGE {page_num + 1} ---\n{page_text.strip()}")
            return "\n\n".join(extracted_text_chunks).strip()
        except Exception as e:
            st.error(f"Error parsing clinical PDF: {e}")
            return ""

def run_chat_page():
    st.markdown(
    """
    <style>
    /* =========================================================
       GLOBAL APP SURFACE BACKGROUND
       ========================================================= */
    .stApp, div[data-testid="stAppViewContainer"] {
        background: #030712 !important;
        color: #f1f5f9 !important;
    }

    /* =========================================================
       FIX: CONTEXT UPLOADER BOX & BORDERED CONTAINERS
       ========================================================= */
    /* Target the exact wrapper Streamlit uses for st.container(border=True) */
    div[data-testid="stVerticalBlockBordered"],
    div[data-testid="stContainerBordered"] {
        background-color: #0f1626 !important;
        background: #0f1626 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    /* Ensure all nested text inside the container card turns light */
    div[data-testid="stVerticalBlockBordered"] *,
    div[data-testid="stContainerBordered"] * {
        color: #f1f5f9 !important;
    }

    /* Target st.file_uploader internal upload zone box */
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #1e293b !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
    }

    /* Fix the "Browse files" button interior */
    div[data-testid="stFileUploaderDropzone"] button {
        background-color: #0f1626 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    /* =========================================================
       FIX: CHAT INPUT STYLING (STICKY BAR & TEXTAREA WRAPPERS)
       ========================================================= */
    /* Target the container that pins the chat input to the bottom */
    div[data-testid="stBottomBlockContainer"] {
        background: #030712 !important;
    }

    /* Target the chat input frame and ALL its inner layout blocks */
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] form {
        background-color: #0f1626 !important;
        background: #0f1626 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
    }

    /* Target the text input space directly */
    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        background-color: transparent !important;
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
        border: none !important;
    }

    /* Style the instruction placeholder text */
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }

    /* =========================================================
       SELECTBOX DROPDOWN POPUPS
       ========================================================= */
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] [role="combobox"] {
        background-color: #0f1626 !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
        color: #f1f5f9 !important;
    }

    div[data-baseweb="popover"] ul, div[role="listbox"] {
        background-color: #0f1626 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    div[role="option"] {
        background-color: transparent !important;
        color: #f1f5f9 !important;
    }

    div[role="option"]:hover {
        background-color: #0ea5e9 !important;
        color: #ffffff !important;
    }

    div[data-testid="stSelectbox"] svg {
        fill: #94a3b8 !important;
        color: #94a3b8 !important;
    }

    /* =========================================================
       EXISTING INTERACTIVE ELEMENT OVERRIDES
       ========================================================= */
    /* Top Row Pill Actions Control Buttons */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 0.4rem 1rem !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    
    div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        color: #ffffff !important;
    }

    /* Isolated Global Chat Message Wrappers */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 1rem 0rem !important;
    }

    /* User Message Bubble Custom Style */
    div[data-testid="stChatMessage"][data-owner="user"] {
        display: flex;
        justify-content: flex-end;
    }
    
    div[data-testid="stChatMessage"][data-owner="user"] > div:nth-child(2) {
        background: linear-gradient(135deg, #094f99 0%, #0c5cb5 100%) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 14px 20px !important;
        max-width: 75% !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
    }

    /* Assistant Card (Clinical Analysis Panel) */
    div[data-testid="stChatMessage"][data-owner="assistant"] > div:nth-child(2) {
        background: rgba(10, 15, 30, 0.7) !important;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4) !important;
    }

    /* Highlighted text accents within clinical response */
    div[data-testid="stChatMessage"][data-owner="assistant"] strong {
        color: #ef4444 !important;
    }

    /* Embedded Multi-Tone Decorative Gradient Block inside results */
    .clinical-gradient-box {
        width: 100%;
        height: 64px;
        background: linear-gradient(90deg, #1d4ed8 0%, #dc2626 50%, #1e3a8a 100%);
        border-radius: 8px;
        margin-top: 16px;
        opacity: 0.85;
    }

    /* Bottom Status/Compliance Meta text styling */
    .chat-metadata-footer {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        color: #475569;
        margin-top: 6px;
        padding: 0 4px;
    }
    
    /* Persistent Navigation Custom Sidebar Fixes */
    .stSidebar {
        background: #060913 !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    # 1. Structural Layout Split (Sidebar Controls vs Main Canvas)
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📄 Context Uploader")
        uploaded_file = st.file_uploader(
            "Upload Generated Clinical Summary Report (PDF)", 
            type=["pdf"],
            help="Drop your downloaded clinical evaluation PDF here to lock down on the patient's record."
        )
        
        # Session State tracking initialization
        if "active_context" not in st.session_state:
            st.session_state.active_context = None
        if "detected_disease" not in st.session_state:
            st.session_state.detected_disease = "General Guidelines Mode"
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Process uploaded file immediately when dropped
        if uploaded_file is not None:
            # Read bytes into memory
            file_bytes = io.BytesIO(uploaded_file.getvalue())
            raw_text = PDFTextExtractor.extract_text_from_stream(file_bytes)
            st.session_state.active_context = raw_text
            
            # Auto-detect disease category from your PDF formatting
            if "Diagnosis Category:" in raw_text:
                # Splits string at target text block and grabs line
                category_line = raw_text.split("Diagnosis Category:")[1].split("\n")[0].strip()
                st.session_state.detected_disease = category_line
            
            st.success("🟢 Patient Profile Locked")
            
            # Clear historical context if an entirely new report is swapped in
            if st.button("Reset Chat Session"):
                st.session_state.messages = []
                # Use safe rerun fallback to handle Streamlit API differences
                try:
                    if hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()
                    elif hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        import streamlit.components.v1 as components
                        components.html("<script>window.location.reload()</script>", height=0)
                        return
                except Exception:
                    import streamlit.components.v1 as components
                    components.html("<script>window.location.reload()</script>", height=0)
                    return
        else:
            st.session_state.active_context = None
            st.session_state.detected_disease = "stroke" # Fallback global reference guideline route
            st.info("🔴 Operating in Global Reference Mode")

        # Visual Context Breakdown Card
        with st.container(border=True):
            st.markdown(f"**Target System Domain:** \n`{st.session_state.detected_disease}`")
            if st.session_state.active_context:
                st.caption(f"Patient Context Size: {len(st.session_state.active_context)} characters")

        # Helper: process and stream a user prompt through the chat engine
        def handle_user_prompt(user_prompt: str):
            if not user_prompt:
                return
            # Display human input block immediately on the canvas
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Log input straight into history
            st.session_state.messages.append({"role": "user", "content": user_prompt})

            # Process and render streamed response blocks
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                try:
                    response_stream = st.session_state.chat_engine.generate_response(
                        user_input=user_prompt,
                        chat_history=st.session_state.messages[:-1],
                        disease_category=st.session_state.detected_disease,
                        local_pdf_text=st.session_state.active_context,
                    )

                    for chunk in response_stream:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as chat_error:
                    st.error(f"Conversational generation workflow interrupted: {chat_error}")
    # 2. Main Chat Feed Rendering Window
    # Display historical turns instantly from memory loop
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat controls: quick prompts + clear chat
    st.markdown("", unsafe_allow_html=True)
    cols = st.columns([1, 1, 1, 3])
    with cols[0]:
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            try:
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
                elif hasattr(st, "rerun"):
                    st.rerun()
                else:
                    import streamlit.components.v1 as components
                    components.html("<script>window.location.reload()</script>", height=0)
                    return
            except Exception:
                import streamlit.components.v1 as components
                components.html("<script>window.location.reload()</script>", height=0)
                return
    with cols[1]:
        if st.button("Summarize Profile"):
            handle_user_prompt("Summarize the patient's profile and key concerns in one paragraph.")
    with cols[2]:
        if st.button("Immediate Actions"):
            handle_user_prompt("List 3 immediate clinical actions based on this patient's data.")

    # Quick prompt selector for custom prompts
    with cols[3]:
        quick = st.selectbox("Quick Prompt:", ["-- choose --", "Suggest further tests", "Give differential diagnoses", "Recommend monitoring plan"], key="quick_prompt")
        if st.button("Use Prompt", key="use_quick_prompt") and quick and quick != "-- choose --":
            mapping = {
                "Suggest further tests": "Suggest further diagnostic tests relevant to this case.",
                "Give differential diagnoses": "Provide a prioritized differential diagnosis list.",
                "Recommend monitoring plan": "Recommend a short monitoring plan and follow-up frequency."
            }
            handle_user_prompt(mapping.get(quick, quick))

    # Accept incoming user prompt inputs via the interaction bar
    if user_prompt := st.chat_input("Ask a follow-up protocol or patient evaluation question..."):
        handle_user_prompt(user_prompt)