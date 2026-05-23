import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv 

# Load environment variables from the root .env file
load_dotenv()

import os
# Force sentence-transformers to use local weights instead of pinging the web
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class MedicalRAGRetriever:
    def __init__(self):
        # Initialize the same 384-dimensional local model used during ingestion
        # It stays in RAM, footprint is small (~150MB), perfectly safe for your system
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def _connect_db(self):
        """Internal helper to access the isolated port 5433 Docker instance."""
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT")
        )

    def retrieve_context(self, disease_category: str, query_text: str, top_k: int = 3) -> list:
        """
        Encodes the search query locally and pulls the top_k most contextually
        relevant clinical reference chunks from the database using cosine similarity.
        """
        # 1. Compute query vector locally (outputs 384 dimensions)
        query_vector = self.embedding_model.encode(query_text).tolist()
        
        conn = self._connect_db()
        cursor = conn.cursor()
        
        # 2. Leverage pgvector's <=> operator (Cosine Distance) for semantic matching.
        # Filter down strictly by disease_category so it doesn't leak cross-domain references.
        search_query = """
        SELECT chunk_text 
        FROM medical_chunks 
        WHERE disease_category = %s 
        ORDER BY embedding <=> %s::vector 
        LIMIT %s;
        """
        
        try:
            cursor.execute(search_query, (disease_category.lower(), query_vector, top_k))
            rows = cursor.fetchall()
            
            # Extract strings from the returned tuples list
            context_chunks = [row[0] for row in rows]
            return context_chunks
            
        except Exception as e:
            print(f"❌ Database Retrieval Error: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

# Simulating isolation testing
if __name__ == "__main__":
    print("🔍 Testing MedicalRAGRetriever against local vector database (port 5433)...")
    
    retriever = MedicalRAGRetriever()
    
    # Test a mock medical prompt search
    test_disease = "stroke"
    test_query = "What should be done if blood pressure or hypertension values are highly elevated?"
    
    matched_chunks = retriever.retrieve_context(disease_category=test_disease, query_text=test_query, top_k=2)
    
    print("\n--- RETRIEVED SEMANTIC DATABASE CHUNKS ---")
    for i, chunk in enumerate(matched_chunks, 1):
        print(f"\n[Match {i}]\n{chunk}")