import psycopg2

def init_database():
    print("🔄 Connecting to fresh PostgreSQL instance...")
    
    # Establish connection to your Docker container
    conn = psycopg2.connect(
        host="localhost",
        database="medical_ai",
        user="postgres",
        password="mysecretpassword",
        port="5432"
    )
    cursor = conn.cursor()
    
    # Drop table if starting completely from scratch to keep data fresh
    cursor.execute("DROP TABLE IF EXISTS medical_chunks;")
    
    # Create the clean production schema for RAG chunks
    create_table_query = """
    CREATE TABLE medical_chunks (
        id SERIAL PRIMARY KEY,
        disease_category VARCHAR(50) NOT NULL,
        chunk_text TEXT NOT NULL
    );
    """
    
    cursor.execute(create_table_query)
    conn.commit()
    print("✅ Fresh 'medical_chunks' table created successfully!")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    init_database()