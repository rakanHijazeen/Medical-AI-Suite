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
        .stSidebar {
            background: rgba(10, 14, 31, 0.95);
            border-right: 1px solid rgba(56,189,248,0.18);
            box-shadow: inset -3px 0 30px rgba(0,0,0,0.22);
        }

        .stSidebar .stMarkdown h3,
        .stSidebar .stMarkdown p,
        .stSidebar .stMarkdown div {
            color: #e2e8f0 !important;
        }

        .stSidebar .stButton>button {
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            color: #ffffff;
            border: none;
            border-radius: 999px;
            padding: 0.95rem 1.2rem;
            font-weight: 700;
            box-shadow: 0 18px 35px rgba(56,189,248,0.18);
        }

        .stFileUploader>div {
            background: rgba(15,23,42,0.92);
            border: 1px dashed rgba(56,189,248,0.45);
            border-radius: 22px;
            padding: 22px;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 24px !important;
            padding: 18px !important;
            margin-bottom: 18px !important;
            box-shadow: 0 16px 40px rgba(0,0,0,0.25) !important;
            border: 1px solid rgba(148,163,184,0.15) !important;
            background: rgba(15,23,42,0.94) !important;
        }

        div[data-testid="stChatMessage"][data-owner="user"] {
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92)) !important;
            border-color: rgba(59,130,246,0.35) !important;
        }

        div[data-testid="stChatMessage"][data-owner="assistant"] {
            background: linear-gradient(135deg, rgba(30,41,59,0.96), rgba(15,23,42,0.90)) !important;
            border-color: rgba(148,163,184,0.22) !important;
        }

        div[data-testid="stChatInput"] {
            border-radius: 999px !important;
            border: 1px solid rgba(148,163,184,0.22) !important;
            background: rgba(15,23,42,0.95) !important;
            padding: 14px !important;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02) !important;
        }

        div[data-testid="stChatInput"] textarea {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
            min-height: 56px !important;
            font-size: 0.95rem !important;
            line-height: 1.4 !important;
            caret-color: #0ea5e9 !important;
        }

        .chat-hero {
            background: linear-gradient(180deg, rgba(14,165,233,0.15), rgba(168,85,247,0.12));
            border: 1px solid rgba(56,189,248,0.22);
            border-radius: 24px;
            padding: 24px 28px;
            margin-bottom: 22px;
            box-shadow: 0 22px 48px rgba(0,0,0,0.22);
        }

        .chat-hero h1 {
            margin: 0;
            color: #f8fafc;
            font-size: 2rem;
            letter-spacing: -0.03em;
        }

        .chat-hero p {
            margin: 10px 0 0;
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.7;
        }

        .chat-controls {
            display: flex;
            gap: 8px;
            align-items: center;
            margin-bottom: 12px;
        }

        .quick-prompt {
            background: rgba(255,255,255,0.04);
            color: #e2e8f0;
            border: 1px solid rgba(255,255,255,0.03);
            padding: 6px 10px;
            border-radius: 999px;
            cursor: pointer;
            font-weight: 600;
        }

        /* Aggressive overrides for Streamlit-generated select/dropdown popups */
        .stSelectbox, .stMultiSelect, .stSelectbox div, .stMultiSelect div,
        .stSelectbox button, .stMultiSelect button,
        .stSelectbox input, .stMultiSelect input,
        .stSelectbox .css-1wy0on6, .stMultiSelect .css-1wy0on6 {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148,163,184,0.14) !important;
        }

        /* listbox / options popup (covers many renderers) */
        [role="listbox"], [role="option"], ul[role="listbox"], li[role="option"], .rc-virtual-list {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
        }

        /* Native select and options */
        select, select option, select:focus, option {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
        }

        /* Remove bright default arrow on some browsers */
        select::-ms-expand { display: none; }

        /* Chat input / textarea overrides (cover multiple generated classes) */
        textarea, .stTextArea textarea, div[data-testid="stChatInput"] textarea, .stChatInput textarea {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148,163,184,0.12) !important;
        }

        /* Ensure dropdown placeholder/text remains visible */
        .stSelectbox .css-1d391kg, .stSelectbox .css-1v3fvcr, .stMultiSelect .css-1d391kg {
            color: #e2e8f0 !important;
        }

        /* Additional overrides to catch chat prompt wrappers and role-based textboxes */
        div[data-testid="stChatInput"] > div, div[data-testid="stChatInput"] > div > div, div[role="textbox"] {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
        }

        /* placeholder color */
        textarea::placeholder, input::placeholder, div[role="textbox"]::placeholder {
            color: #94a3b8 !important;
        }
        /* Force dark backgrounds and high-contrast text for all form controls */
        select, option, input, textarea, .stSelectbox, .stTextInput, .stNumberInput, .stMultiSelect {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148,163,184,0.14) !important;
        }

        /* Specific overrides for select dropdowns and their options */
        select, .stSelectbox select, .stSelectbox div[role="combobox"] input {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
        }
        option {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
        }

        /* Tighter padding for inputs inside forms */
        input[type="text"], input[type="number"], textarea {
            padding: 10px !important;
            border-radius: 8px !important;
        }

        /* Target Streamlit BaseWeb select component and all descendants */
        [data-baseweb="select"],
        [data-baseweb="select"] * {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
        }

        /* Combobox input specifically */
        input[role="combobox"],
        input[role="combobox"]:focus {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148,163,184,0.14) !important;
        }

        /* Listbox popup when opened */
        [role="listbox"], [role="listbox"] * {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
        }

        /* Option items in listbox */
        [role="option"] {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
        }
        [role="option"]:hover {
            background: rgba(56,189,248,0.12) !important;
            color: #e2e8f0 !important;
        }

        /* Specific overrides for chat input textarea */
        [data-baseweb="textarea"],
        [data-baseweb="base-input"],
        textarea[data-testid="stChatInputTextArea"] {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148,163,184,0.12) !important;
        }

        textarea[data-testid="stChatInputTextArea"]::placeholder {
            color: #94a3b8 !important;
        }

        /* Force dark on entire chat input container and all children */
        div[data-testid="stChatInput"],
        div[data-testid="stChatInput"] * {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
        }

        /* Ensure the textarea text input is dark and readable */
        div[data-baseweb="textarea"] textarea,
        textarea {
            background: rgba(15,23,42,0.96) !important;
            color: #e2e8f0 !important;
            caret-color: #0ea5e9 !important;
        }

        /* Force dark on outer chat input container wrappers */
        .stBottom,
        .stElementContainer,
        .stVerticalBlock,
        div[data-testid="stBottomBlockContainer"],
        div[data-testid="stVerticalBlock"],
        div[data-testid="stElementContainer"] {
            background: rgba(15,23,42,0.96) !important;
        }

        /* Chat input field - fix border edges */
        div[data-testid="stChatInput"] {
            border: 1px solid rgba(56,189,248,0.25) !important;
            border-radius: 12px !important;
            background: rgba(20,28,50,0.8) !important;
            box-shadow: inset 0 0 0 1px rgba(56,189,248,0.08) !important;
        }

        /* Sidebar input fields - remove white backgrounds */
        .stSidebar input[type="text"],
        .stSidebar .stTextInput input,
        .stSidebar div[data-baseweb="input"] input,
        .stSidebar input {
            background: rgba(20,28,50,0.8) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148,163,184,0.16) !important;
            border-radius: 8px !important;
        }

        .stSidebar div[data-baseweb="input"] {
            background: rgba(20,28,50,0.8) !important;
            border: 1px solid rgba(148,163,184,0.16) !important;
        }

        /* Sidebar select field backgrounds */
        .stSidebar .stSelectbox,
        .stSidebar [data-baseweb="select"] {
            background: rgba(20,28,50,0.8) !important;
            border: 1px solid rgba(148,163,184,0.16) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="chat-hero"><h1>💬 Interactive Case Consultant Workspace</h1><p>Query global clinical reference guidelines alongside localized patient audit records.</p></div>',
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