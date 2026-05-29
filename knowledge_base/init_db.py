import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def init_database():
    print("🔄 Connecting to target PostgreSQL database instance...")
    
    # Track configurations dynamically using environment variables
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "medical_ai"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "mysecretpassword"),
        port=os.getenv("DB_PORT", "5433")
    )
    cursor = conn.cursor()
    
    # Purge old structures
    cursor.execute("DROP TABLE IF EXISTS medical_chunks;")
    
    # Create the modern production schema matching the dual-text RAG pipeline
    create_table_query = """
    -- 1. Explicitly activate the vector extension in the database instance
    CREATE EXTENSION IF NOT EXISTS vector;

    -- 2. Create the data schema tracking categories, raw text, and vector spaces
    CREATE TABLE medical_chunks (
        id SERIAL PRIMARY KEY,
        disease_category VARCHAR(50) NOT NULL, -- 'heart', 'stroke', 'kidney', 'diabetes'
        chunk_text TEXT NOT NULL,              -- Formatted with metadata headers for Llama 3.1
        clean_text TEXT NOT NULL,              -- Clean text body strictly for sentence embeddings
        embedding VECTOR(384) NOT NULL         -- 384 dimensions matching all-MiniLM-L6-v2
    );
    """
    
    print("🏗️ Creating production table schemas...")
    cursor.execute(create_table_query)
    conn.commit()
    print("✅ Fresh 'medical_chunks' table structure initialized successfully!")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    init_database()