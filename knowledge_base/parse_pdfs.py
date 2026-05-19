import os
import psycopg2
import pdfplumber

def connect_db():
    return psycopg2.connect(
        host="localhost",
        database="medical_ai",
        user="postgres",
        password="mysecretpassword",
        port="5432"
    )

def convert_table_to_markdown(table_data) -> str:
    """Converts a raw list-of-lists table into a clean Markdown table string."""
    if not table_data or not table_data[0]:
        return ""
    
    # Clean up None values and strip whitespace from cells
    cleaned_table = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table_data]
    
    # Build the header row
    headers = cleaned_table[0]
    markdown_table = "| " + " | ".join(headers) + " |\n"
    # Build the separator line (e.g., |---|---|)
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Build the data rows
    for row in cleaned_table[1:]:
        # Skip purely empty rows
        if any(cell for cell in row):
            markdown_table += "| " + " | ".join(row) + " |\n"
            
    return markdown_table + "\n"

def extract_layout_aware_chunks(pdf_path: str) -> list:
    """
    Extracts content page-by-page. Tables are converted to Markdown grids,
    and narrative text is kept intact, preventing layout scrambling.
    """
    chunks = []
    
    print(f"📖 Opening complex PDF: {os.path.basename(pdf_path)}")
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # 1. Find and extract tables on this page
            tables = page.extract_tables()
            table_text_blocks = []
            
            for table in tables:
                md_table = convert_table_to_markdown(table)
                if md_table:
                    table_text_blocks.append(md_table)
            
            # 2. Extract standard text lines, skipping things already caught in tables
            # (pdfplumber allows us to get clean text layouts)
            raw_text = page.extract_text()
            
            # 3. Combine structural elements into clean, standalone chunks for this page
            if table_text_blocks:
                # If page has tables, we save the tables as independent, high-priority chunks
                for md_table in table_text_blocks:
                    chunks.append(f"[TABLE DATA - PAGE {page_num}]\n{md_table}")
            
            if raw_text:
                # Clean up general text narrative blocks
                clean_text = raw_text.strip()
                if len(clean_text) > 40: # Ignore tiny artifacts, headers, or footers
                    chunks.append(f"[CLINICAL NOTES - PAGE {page_num}]\n{clean_text}")
                    
    return chunks

def insert_chunks_to_db(disease_category: str, chunks: list):
    conn = connect_db()
    cursor = conn.cursor()
    
    insert_query = "INSERT INTO medical_chunks (disease_category, chunk_text) VALUES (%s, %s);"
    
    try:
        for chunk in chunks:
            cursor.execute(insert_query, (disease_category.lower(), chunk))
        conn.commit()
        print(f"✅ Successfully ingested {len(chunks)} structural layout chunks for: '{disease_category}'")
    except Exception as e:
        conn.rollback()
        print(f"❌ Database Insertion Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Ensure data folder exists
    os.makedirs("knowledge_base/data", exist_ok=True)
    
    # Map your target files to categories
    file_pipeline = {
        "stroke_guidelines.pdf": "stroke",
        # "kidney_guidelines.pdf": "kidney",
        # "heart_guidelines.pdf": "heart"
    }
    
    print("🚀 Initiating Local Layout-Aware Ingestion Engine...")
    
    for filename, category in file_pipeline.items():
        pdf_path = os.path.join("knowledge_base/data", filename)
        
        if os.path.exists(pdf_path):
            # Run the extraction layout pipeline
            processed_chunks = extract_layout_aware_chunks(pdf_path)
            # Push live to Docker Postgres
            insert_chunks_to_db(disease_category=category, chunks=processed_chunks)
        else:
            print(f"⚠️  No file found at: {pdf_path}")
            print("👉 Drop your complex medical PDFs into 'knowledge_base/data/' to process them structurally!")