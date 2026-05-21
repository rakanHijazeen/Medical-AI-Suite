# test_rag.py (Make sure this sits right in your root Medical_AI_Suite folder!)
import os
import sys
from pathlib import Path

# Add the current root directory to the Python path to eliminate import traps
sys.path.append(str(Path(__file__).resolve().parent))

from app.rag import generate_medical_audit

if __name__ == "__main__":
    print("🔥 Initiating full pipeline integration test (Postgres 5433 -> Groq API)...")
    
    # 1. Mock parameters matching your ML classifier outputs
    disease_input = "stroke"
    risk_input = 0.842
    metrics_input = {"age": 67, "hypertension": 1, "avg_glucose_level": 228.1}
    
    # 2. Fire the complete execution cycle
    try:
        final_report = generate_medical_audit(
            disease=disease_input,
            risk_score=risk_input,
            patient_metrics=metrics_input
        )
        
        print("\n================= GROQ AUDIT REPORT =================")
        print(final_report)
        print("=====================================================")
        
    except Exception as e:
        print(f"❌ Integration Test Failed: {e}")