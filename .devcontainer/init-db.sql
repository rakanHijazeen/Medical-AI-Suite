-- Medical AI Suite Database Initialization
-- This script is executed when PostgreSQL container starts

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS medical_ai;

-- Create tables for audit logging
CREATE TABLE IF NOT EXISTS medical_ai.evaluation_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    patient_id VARCHAR(255),
    condition VARCHAR(100),
    model_prediction FLOAT,
    risk_level VARCHAR(50),
    audit_status VARCHAR(50),
    llm_response TEXT,
    audit_comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_evaluation_logs_timestamp 
    ON medical_ai.evaluation_logs(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_evaluation_logs_patient_id 
    ON medical_ai.evaluation_logs(patient_id);

CREATE INDEX IF NOT EXISTS idx_evaluation_logs_condition 
    ON medical_ai.evaluation_logs(condition);

-- Create table for vector embeddings (for RAG)
CREATE TABLE IF NOT EXISTS medical_ai.knowledge_base (
    id SERIAL PRIMARY KEY,
    source_doc VARCHAR(255),
    chunk_text TEXT,
    embedding_vector FLOAT8[],  -- pgvector type if available
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_knowledge_base_source 
    ON medical_ai.knowledge_base(source_doc);

-- Create audit trail table
CREATE TABLE IF NOT EXISTS medical_ai.audit_trail (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100),
    user_action VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);

CREATE INDEX IF NOT EXISTS idx_audit_trail_timestamp 
    ON medical_ai.audit_trail(timestamp DESC);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA medical_ai TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA medical_ai TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA medical_ai TO postgres;

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "json1" WITH SCHEMA public;

-- Log successful initialization
INSERT INTO medical_ai.audit_trail (event_type, user_action, details)
VALUES ('DATABASE', 'INITIALIZATION', '{"status": "success", "timestamp": "' || NOW() || '"}');
