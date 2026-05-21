import psycopg2

def init_database():
    print("🔄 Connecting to fresh PostgreSQL instance...")
    
    # Establish connection to your Docker container
    conn = psycopg2.connect(
        host="localhost",
        database="medical_ai",
        user="postgres",
        password="mysecretpassword",
        port="5433"  # Kept on port 5433 to completely isolate from Notatk
    )
    cursor = conn.cursor()
    
    # Drop table if starting completely from scratch to keep data fresh
    cursor.execute("DROP TABLE IF EXISTS medical_chunks;")
    
    # Create the clean production schema for RAG chunks with pgvector support
    create_table_query = """
    -- 1. Explicitly activate the vector extension in the database instance
    CREATE EXTENSION IF NOT EXISTS vector;

    -- 2. Create the data schema tracking categories, raw text, and vector spaces
    CREATE TABLE medical_chunks (
        id SERIAL PRIMARY KEY,
        disease_category VARCHAR(50) NOT NULL, -- 'heart', 'stroke', 'kidney'
        chunk_text TEXT NOT NULL,
        embedding VECTOR(1536) NOT NULL        -- 1536 dimensions for standard OpenAI/cohere embeddings
    );

    -- 3. Create a fast vector lookup index for your retrieval search queries
    CREATE INDEX IF NOT EXISTS medical_chunks_vector_idx 
    ON medical_chunks USING hnsw (embedding vector_cosine_ops);
    """
    
    cursor.execute(create_table_query)
    conn.commit()
    print("✅ Fresh 'medical_chunks' table with vector capabilities created successfully!")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    init_database()