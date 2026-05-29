from .retriever import MedicalRAGRetriever
from .engine import MedicalRAGEngine

# Global cache variables for singleton instances
_RETRIEVER_INSTANCE = None
_ENGINE_INSTANCE = None

def _get_retriever():
    """Lazy-loads the retriever instance once and keeps it in memory."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = MedicalRAGRetriever()
    return _RETRIEVER_INSTANCE

def _get_engine():
    """Lazy-loads the engine instance once and keeps it in memory."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = MedicalRAGEngine()
    return _ENGINE_INSTANCE

def generate_medical_audit(disease: str, risk_score: float, patient_metrics: dict) -> str:
    """
    Orchestrates the clean RAG pipeline lifecycle using optimized, 
    natural-language semantic search extraction.
    """
    # Use cached singletons to prevent memory leaks and slow initializations
    retriever = _get_retriever()
    engine = _get_engine()
    
    # 1. Convert raw patient data dictionary into a clean, semantic prose sentence
    metric_conditions = [f"elevated {k}" for k, v in patient_metrics.items() if v == 1 or (isinstance(v, (int, float)) and v > 100)]
    
    if metric_conditions:
        symptoms_str = " with " + " and ".join(metric_conditions)
    else:
        symptoms_str = ""

    # 2. Build a clean sentence optimized for matching PDF documentation structures
    search_query = f"Clinical guidelines and treatment protocol for {disease}{symptoms_str}."
    
    # Executing the vector lookup
    context_chunks = retriever.retrieve_context(
        disease_category=disease, 
        query_text=search_query, 
        top_k=3
    )
    
    # Executing the double prompt processing chain
    report = engine.generate_clinical_explanation(
        disease=disease,
        risk_score=risk_score,
        patient_metrics=patient_metrics,
        context_chunks=context_chunks
    )
    
    return report