import os
import psycopg2
from pypdf import PdfReader
from pathlib import Path
from sentence_transformers import SentenceTransformer

print("⏳ Loading local embedding engine (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding engine loaded successfully!")

def connect_db():
    return psycopg2.connect(
        host="localhost",
        database="medical_ai",
        user="postgres",
        password="mysecretpassword",
        port="5433"
    )

def create_overlapping_chunks(text: str, category: str, page_num: int, window_size=500, overlap=150) -> list:
    """
    Slices raw text into overlapping windows. This ensures that fragmented text from
    flowcharts and tables retains context from nearby headers and descriptions.
    """
    words = text.split()
    chunks = []
    
    if len(words) <= window_size:
        clean_text = f"[CONTEXT: {category.upper()} | PAGE: {page_num}]\n{text.strip()}"
        return [clean_text]
        
    i = 0
    while i < len(words):
        # Create a slice of words
        word_slice = words[i:i + window_size]
        slice_text = " ".join(word_slice)
        
        # Format with rich context headers so the vector model knows what it's looking at
        chunk_entry = f"[CONTEXT: {category.upper()} | PAGE: {page_num} | SEGMENT: {len(chunks)+1}]\n{slice_text.strip()}"
        chunks.append(chunk_entry)
        
        # Move forward by window size minus the overlap
        i += (window_size - overlap)
        
    return chunks

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
    print(f"🔍 Searching inside: {REFERENCE_DIR}\n")
    
    conn = connect_db()
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO medical_chunks (disease_category, chunk_text, embedding) 
    VALUES (%s, %s, %s);
    """
    
    BATCH_SIZE = 32
    
    try:
        for filename, category in file_pipeline.items():
            pdf_path = REFERENCE_DIR / filename
            
            if pdf_path.exists():
                print(f"📖 Parsing text layers & building context windows for: {filename}")
                reader = PdfReader(pdf_path)
                
                all_file_chunks = []
                
                # 1. Process page-by-page to catch dense tables and flowcharts
                for page_num, page in enumerate(reader.pages, 1):
                    raw_text = page.extract_text()
                    if raw_text and len(raw_text.strip()) > 50:
                        # Build context-stamped windows for this page
                        page_chunks = create_overlapping_chunks(raw_text, category, page_num)
                        all_file_chunks.extend(page_chunks)
                
                total_chunks = len(all_file_chunks)
                print(f"📊 Generated {total_chunks} contextual chunks. Saving to database...")
                
                # 2. Bulk process vectors in safe, rapid batches
                for i in range(0, total_chunks, BATCH_SIZE):
                    batch_chunks = all_file_chunks[i:i + BATCH_SIZE]
                    batch_vectors = embedding_model.encode(batch_chunks, show_progress_bar=False).tolist()
                    
                    for chunk, chunk_vector in zip(batch_chunks, batch_vectors):
                        cursor.execute(insert_query, (category.lower(), chunk, chunk_vector))
                    
                    conn.commit()
                    progress = min(i + BATCH_SIZE, total_chunks)
                    print(f"   📥 Progress: {progress}/{total_chunks} segments indexed...")
                    
                print(f"✅ Successfully finished ingestion for: '{category}'!\n")
            else:
                print(f"⚠️  No file found at: {pdf_path}")
                
    except Exception as e:
        conn.rollback()
        print(f"❌ Pipeline Execution Error: {e}")
    finally:
        cursor.close()
        conn.close()