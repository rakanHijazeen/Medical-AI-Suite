import os
import json
import psycopg2
from utils.logger import logger  # Imports system log framework for unexpected dropouts

def get_db_connection():
    """Builds an isolated database connection instance using protected environment keys."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "medical_ai"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5433")
    )

def log_evaluation(category: str, score: float, metrics: dict, chunks: list, explanation: str, report: str, hallucination: bool):
    """
    Safely commits a production clinical evaluation record into PostgreSQL.
    Wraps anomalies safely to prevent database transaction state corruption from crashing the API.
    """
    query = """
    INSERT INTO evaluation_logs (
        disease_category, risk_score, patient_metrics, 
        retrieved_chunks, generated_explanation, audited_report, is_hallucination_detected
    ) VALUES (%s, %s, %s, %s, %s, %s, %s);
    """
    
    try:
        # Enforcing connection boundaries via context managers loops safety routines natively
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (
                    category.lower(),
                    score,
                    json.dumps(metrics),  # Serializes python dictionary maps straight into Postgres JSONB
                    chunks,              # Psycopg2 maps Python lists natively to TEXT[] database fields
                    explanation,
                    report,
                    hallucination
                ))
                conn.commit()
    except Exception as e:
        # Crucial Catch: If your tracking log table database operation fails,
        # flag a serious warning in app.log, but DO NOT drop or crash the active Streamlit app thread!
        logger.error(f"PostgreSQL Operational Fail: Unable to persist metrics tracking payload. Reason: {e}")