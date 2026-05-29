import os
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv 

load_dotenv()

# Prevent unnecessary internet pings on production environments
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class MedicalRAGRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def _connect_db(self):
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT")
        )

    def retrieve_context(self, disease_category: str, query_text: str, top_k: int = 3) -> list:
        """
        Encodes search queries, handles pgvector cosine matching, and sanitizes 
        ingestion metadata headers before sending context data to the LLM pipeline.
        """
        # 1. Compute query vector locally
        query_vector = self.embedding_model.encode(query_text).tolist()
        
        conn = None
        cursor = None
        context_chunks = []
        
        # 2. Query execution using Cosine Distance (<=>)
        search_query = """
        SELECT chunk_text 
        FROM medical_chunks 
        WHERE disease_category = %s 
        ORDER BY embedding <=> %s::vector 
        LIMIT %s;
        """
        
        try:
            conn = self._connect_db()
            cursor = conn.cursor()
            
            cursor.execute(search_query, (disease_category.lower(), query_vector, top_k))
            rows = cursor.fetchall()
            
            for row in rows:
                raw_text = row[0]
                
                # Clean up old ingestion headers so they do not confuse Llama 3.1
                if "\n" in raw_text and raw_text.startswith("[CONTEXT:"):
                    # Splits at the first newline character, discarding the raw metadata bracket
                    clean_text = raw_text.split("\n", 1)[1].strip()
                else:
                    clean_text = raw_text.strip()
                    
                context_chunks.append(clean_text)
                
            return context_chunks
            
        except Exception as e:
            print(f"❌ Database Retrieval Error: {e}")
            return []
        finally:
            # Safe closing pattern preventing UnboundLocalErrors
            if cursor:
                cursor.close()
            if conn:
                conn.close()

if __name__ == "__main__":
    print("🔍 Testing MedicalRAGRetriever against local vector database...")
    retriever = MedicalRAGRetriever()
    
    test_disease = "stroke"
    test_query = "Protocol for calculated risk exceeding 80 percent and immediate blood pressure control."
    
    matched_chunks = retriever.retrieve_context(disease_category=test_disease, query_text=test_query, top_k=2)
    
    print("\n--- SANITIZED RETRIEVED MEDICAL CHUNKS ---")
    for i, chunk in enumerate(matched_chunks, 1):
        print(f"\n[Clean Match {i}]\n{chunk}")