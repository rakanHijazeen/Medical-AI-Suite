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
    st.title("💬 Interactive Case Consultant Workspace")
    st.caption("Query global clinical reference guidelines alongside localized patient audit records.")

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
                st.rerun()
        else:
            st.session_state.active_context = None
            st.session_state.detected_disease = "stroke" # Fallback global reference guideline route
            st.info("🔴 Operating in Global Reference Mode")

        # Visual Context Breakdown Card
        with st.container(border=True):
            st.markdown(f"**Target System Domain:** \n`{st.session_state.detected_disease}`")
            if st.session_state.active_context:
                st.caption(f"Patient Context Size: {len(st.session_state.active_context)} characters")

    # 2. Main Chat Feed Rendering Window
    # Display historical turns instantly from memory loop
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept incoming user prompt inputs via the interaction bar
    if user_prompt := st.chat_input("Ask a follow-up protocol or patient evaluation question..."):
        # Display human input block immediately on the canvas
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        # Log input straight into history
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Process and render streamed response blocks
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Connect execution directly to your chat engine generation loop
            try:
                response_stream = st.session_state.chat_engine.generate_response(
                    user_input=user_prompt,
                    chat_history=st.session_state.messages[:-1], # Send previous conversation structure
                    disease_category=st.session_state.detected_disease,
                    local_pdf_text=st.session_state.active_context
                )
                
                # Iterate on incoming token chunks via yield statements
                for chunk in response_stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                # Lock final response view without typing cursor
                response_placeholder.markdown(full_response)
                
                # Append assistant execution block into permanent session memory array
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as chat_error:
                st.error(f"Conversational generation workflow interrupted: {chat_error}")