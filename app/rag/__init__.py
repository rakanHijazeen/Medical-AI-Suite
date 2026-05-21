# app/__init__.py
from .retriever import MedicalRAGRetriever
from .engine import MedicalRAGEngine

def generate_medical_audit(disease: str, risk_score: float, patient_metrics: dict) -> str:
    # We instantiate them INSIDE the function to prevent boot-up import loops
    retriever = MedicalRAGRetriever()
    engine = MedicalRAGEngine()
    
    search_query = f"Risk factors, clinical interventions, and guidelines for {disease}. Patient data: {patient_metrics}"
    
    context_chunks = retriever.retrieve_context(
        disease_category=disease, 
        query_text=search_query, 
        top_k=3
    )
    
    report = engine.generate_clinical_explanation(
        disease=disease,
        risk_score=risk_score,
        metrics=patient_metrics,
        context_chunks=context_chunks
    )
    
    return report