import os
import io
import time
import logging
import groq
from groq import Groq
from dotenv import load_dotenv
from app.rag.retriever import MedicalRAGRetriever
from pypdf import PdfReader
from utils.logger import logger

load_dotenv()

LOG_PATH = os.path.join("logs", "chat.log")

os.makedirs("logs", exist_ok=True)

chat_file_handler = logging.FileHandler(LOG_PATH)
chat_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(chat_file_handler)

class MedicalChatEngine:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.retriever = MedicalRAGRetriever()
        
        self.primary_model = "llama-3.1-8b-instant" # for response generation
        self.helper_model = "llama-3.3-70b-versatile"    # for lightweight query rewriting
        logger.info(f"MedicalChatEngine initialized successfully | Orchestrator: {self.helper_model} -> {self.primary_model}")
    
    def _optimize_query(self, history, user_input):
        """
        Rewrites the user query to be standalone by considering 
        the previous chat history.
        """
        # If no history, just return the input
        if not history:
            return user_input
        
        logger.info(f"Optimizing multi-turn context payload using {self.helper_model}...")

        context_string = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
        
        prompt = f"""
        Given the following conversation history and a follow-up question, 
        rephrase the follow-up question to be a standalone search query.
        
        History:
        {context_string}
        
        Follow-up: {user_input}
        Standalone Query:"""

        response = self.client.chat.completions.create(
            model=self.helper_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        # time.sleep(1)  # Add a small delay to avoid rate limiting
        optimized_query = response.choices[0].message.content.strip()
        logger.info(f"Optimized search query generated | Query: '{optimized_query}'")
        return optimized_query

    def generate_response(self, user_input, chat_history, disease_category, local_pdf_text=None):
        """
        Orchestrates retrieval and generates a streamed response.
        """

        logger.info(f"Incoming conversational RAG prompt received | History Depth: {len(chat_history)} turns")

        def safe_groq_call(self, payload):
            for attempt in range(3): # Try 3 times
                try:
                    return self.client.chat.completions.create(**payload)
                except groq.RateLimitError:
                    time.sleep(2) # Wait 2 seconds before trying again
                    continue
            return None

        # 1. Optimize the query for vector search
        search_query = self._optimize_query(chat_history, user_input)

        # 2. Retrieve Global Context (Guidelines)
        # Using your existing retrieve_context method from retriever.py
        logger.info(f"Executing pgvector similarity search against {disease_category} guidelines...")
        global_chunks = self.retriever.retrieve_context(
            disease_category=disease_category, 
            query_text=search_query, 
            top_k=3
        )
        global_context = "\n\n".join(global_chunks)

        logger.info(f"Vector search complete | Context payload size: {len(global_context)} characters")

        # 3. Construct System Prompt
        system_prompt = f"""
        You are an expert Clinical Decision Support AI. Your goal is to assist a clinician 
        based on TWO specific sources of truth:
        
        1. LOCAL PATIENT DATA (Extracted from their report):
        {local_pdf_text if local_pdf_text else "No specific patient report uploaded."}
        
        2. GLOBAL MEDICAL GUIDELINES (Retrieved from Database):
        {global_context}
        
        STRICT RULES:
        - USER CLAIMS VERIFICATION: If the user makes a factual claim about what is written inside the uploaded PDF report (e.g., "The report says X"), verify it against the `local_pdf_text`. If the claim is factually false, professionally correct them.
        - CLINICAL REASONING ALLOWED: Do not be limited to only repeating what is in the PDF. You are explicitly expected to interpret the patient's metrics using your clinical knowledge and the retrieved global guidelines.
        - If the patient report contradicts the general guidelines, highlight the discrepancy.
        - Use a professional, clinical tone.
        - If the answer is not in the provided context, state that you do not have enough information.
        - Keep answers concise and evidence-based.
        """

        # 4. Prepare Message Payload
        messages = [{"role": "system", "content": system_prompt}]
        # Append sliding history (last 5 turns to save tokens)
        messages.extend(chat_history[-5:])
        messages.append({"role": "user", "content": user_input})

        # 5. Stream Completion
        logger.info(f"Assembling context payloads and initiating live RAG stream via {self.primary_model}...")
        completion = self.client.chat.completions.create(
            model=self.primary_model, # Use the more powerful model (llama-3.1-8b-instant)
            messages=messages,
            temperature=0.0,
            stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

        # Log completion of the stream
        logger.info("Conversational RAG response stream completed successfully.")

class PDFTextExtractor:
    @staticmethod
    def extract_text_from_stream(file_stream: io.BytesIO) -> str:
        """
        Takes a raw binary PDF stream from Streamlit's file uploader,
        reads its pages sequentially, and extracts clean, raw text blocks.
        """
        try:
            # 1. Initialize the reader directly from the byte stream
            reader = PdfReader(file_stream)
            extracted_text_chunks = []
            
            # 2. Iterate through all document pages
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_text_chunks.append(f"--- PAGE {page_num + 1} --- \n{page_text.strip()}")
            
            # 3. Join blocks into a single structured string
            full_report_text = "\n\n".join(extracted_text_chunks)
            logger.info(f"PDF extraction complete | Pages processed: {len(reader.pages)}")
            return full_report_text.strip()
            
        except Exception as e:
            # Safe clinical fallback preventing app crashes if an invalid file is dropped
            logger.error(f"PDF Text Extraction Error: {str(e)}")
            return ""                