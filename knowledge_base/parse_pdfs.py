import os
import psycopg2
from pypdf import PdfReader
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Fixed maximum sequence length safely under MiniLM's token saturation point
TOKEN_LIMIT = 200  
OVERLAP_LIMIT = 50

print("⏳ Loading local embedding engine (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Safeguard to enforce structural truncation inside the model
embedding_model.max_seq_length = 256
print("✅ Embedding engine loaded successfully!")

def connect_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )

def generate_semantic_chunks(full_text: str, window_size=TOKEN_LIMIT, overlap=OVERLAP_LIMIT) -> list:
    """
    Splits text accurately by counting structural word units.
    Separates embedding content from prompt display blocks.
    """
    words = full_text.split()
    chunks_records = []
    
    i = 0
    segment_idx = 1
    while i < len(words):
        word_slice = words[i:i + window_size]
        slice_text = " ".join(word_slice).strip()
        
        if len(slice_text) > 30: # Ignore noise blocks
            chunks_records.append({
                "clean_text": slice_text,
                "segment_id": segment_idx
            })
            segment_idx += 1
            
        i += (window_size - overlap)
        
    return chunks_records

if __name__ == "__main__":
    current_path = Path(__file__).resolve()
    root_dir = current_path
    for parent in current_path.parents:
        if parent.name == "Medical_AI_Suite":
            root_dir = parent
            break
            
    REFERENCE_DIR = root_dir / "data" / "medical_reference"
    
    file_pipeline = {
        "stroke_guidelines.pdf": "stroke",
        "kidney_guidelines.pdf": "kidney",
        "heart_guidelines.pdf": "heart",
        "diabetes_guidelines.pdf": "diabetes"
    }
    
    print("\n🚀 Initiating Context-Aware Ingestion Engine...")
    conn = connect_db()
    cursor = conn.cursor()
    
    # Clean old broken indexes
    print("🧹 Purging old broken vector indexes...")
    cursor.execute("TRUNCATE TABLE medical_chunks;")
    conn.commit()
    
    # Multi-column schema mapping clean text vs embedding context
    insert_query = """
    INSERT INTO medical_chunks (disease_category, chunk_text, clean_text, embedding) 
    VALUES (%s, %s, %s, %s);
    """
    
    BATCH_SIZE = 32
    
    try:
        for filename, category in file_pipeline.items():
            pdf_path = REFERENCE_DIR / filename
            
            if pdf_path.exists():
                print(f"📖 Reading layout structures for: {filename}")
                reader = PdfReader(pdf_path)
                
                # Merge document stream to prevent page split fragmentation
                full_document_text = []
                page_mappings = [] # Keep track of page offsets dynamically
                
                for page_num, page in enumerate(reader.pages, 1):
                    raw_text = page.extract_text()
                    if raw_text:
                        full_document_text.append(raw_text)
                
                compiled_text = "\n".join(full_document_text)
                structural_chunks = generate_semantic_chunks(compiled_text)
                
                total_chunks = len(structural_chunks)
                print(f"📊 Extracted {total_chunks} safe chunks. Encoding vectors...")
                
                for i in range(0, total_chunks, BATCH_SIZE):
                    batch = structural_chunks[i:i + BATCH_SIZE]
                    
                    # Extract raw texts for the embedding calculation
                    texts_to_embed = [item["clean_text"] for item in batch]
                    batch_vectors = embedding_model.encode(texts_to_embed, show_progress_bar=False).tolist()
                    
                    for item, vector in zip(batch, batch_vectors):
                        clean_body = item["clean_text"]
                        # Format the text with metadata context strictly for Llama 3.1 consumption
                        formatted_llm_text = f"[SOURCE: {category.upper()} | CHUNK: {item['segment_id']}]\n{clean_body}"
                        
                        cursor.execute(insert_query, (category.lower(), formatted_llm_text, clean_body, vector))
                    
                    conn.commit()
                    
                print(f"✅ Indexed standard references for: '{category}'\n")
            else:
                print(f"⚠️ No file found at: {pdf_path}")
        # =====================================================================
        # ⚡ HNSW INDEX BUILD ZONE (RUNS ONCE AFTER ALL INSERTS COMPLETE)
        # =====================================================================
        print("⚡ Generating fast look-up HNSW vector graphs across all rows...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS medical_chunks_vector_idx 
            ON medical_chunks USING hnsw (embedding vector_cosine_ops);
        """)
        conn.commit()
        print("🚀 HNSW Index built successfully! Your data is fully synchronized.")   
             
    except Exception as e:
        conn.rollback()
        print(f"❌ Pipeline Execution Error: {e}")
    finally:
        cursor.close()
        conn.close()