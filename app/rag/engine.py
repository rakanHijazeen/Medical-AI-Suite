import os 
import psycopg2 # Added to handle DB connection check
from groq import Groq 
from utils.logger import logger  # 👈 System text log stream
from utils.audit_logger import log_evaluation  # 👈 Postgres evaluation logger

class MedicalRAGEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            logger.critical("Engine initialization failed: Missing GROQ_API_KEY environment variable.")
            raise ValueError("❌ Missing Groq API Key. Please set it in your terminal environment.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant"
        logger.info(f"MedicalRAGEngine initialized successfully using core model: {self.model}")

    def generate_clinical_explanation(self, disease: str, risk_score: float, metrics: dict, context_chunks: list) -> str:
        formatted_context = "\n\n".join([f"- {chunk}" for chunk in context_chunks])
        
        system_prompt = (
            "You are an expert clinical AI assistant. Rely ONLY on the provided "
            "Medical Reference Guidelines context. Do not assume or extrapolate facts. "
            "Do not include conversational fluff. Use clean Markdown formatting."
        )
        
        user_prompt = f"""
### PATIENT CLINICAL DATA
- **Target Disease Assessment:** {disease.upper()}
- **Calculated Risk Score:** {risk_score * 100:.1f}%
- **Patient Metrics Evaluated:** {metrics}

### MEDICAL REFERENCE GUIDELINES (Retrieved Context)
{formatted_context if formatted_context else "No reference guidelines found."}
"""

        try:
            # --- STAGE 1: LOG GENERATION INITIATION ---
            logger.info(f"Sending Stage 1 prompt to Groq for disease framework: {disease.upper()}")
            
            gen_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            initial_explanation = gen_response.choices[0].message.content
            logger.info("Stage 1 raw clinical explanation generated.")

            # =====================================================================
            # STAGE 2: THE GUARD / AUDITOR LLM
            # =====================================================================
            logger.info("Passing output to Stage 2 Auditor LLM for clinical compliance validation...")
            
            guard_system_prompt = (
                "You are a strict Medical AI Auditor and Medical Fact-Checker. Your sole job is to cross-examine "
                "a generated clinical explanation against the raw source Medical Reference Guidelines.\n\n"
                "CRITICAL RULES:\n"
                "1. If the explanation contains ANY facts, medications, or clinical benchmarks NOT explicitly mentioned "
                "in the source guidelines, you must flag it and strip out the hallucination.\n"
                "2. Ensure the tone is objective and completely safe for clinical decision support.\n"
                "3. Output a finalized, audited version of the text. Do not include conversational remarks like 'I have audited this'."
            )

            guard_user_prompt = f"""
### RAW SOURCE MEDICAL GUIDELINES (THE TRUTH)
{formatted_context}

### GENERATED CLINICAL EXPLANATION TO AUDIT
{initial_explanation}

---
Review the explanation carefully. Output the finalized, verified explanation below. If changes or deletions are necessary for absolute accuracy to the truth data, apply them directly.
"""

            audit_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": guard_system_prompt},
                    {"role": "user", "content": guard_user_prompt}
                ],
                temperature=0.1, 
                max_tokens=1024
            )
            
            final_audited_report = audit_response.choices[0].message.content
            logger.info("Stage 2 clinical audit evaluation complete.")

            # --- STAGE 3: TELEMETRY EVALUATION LOGGING ---
            # Simple heuristic: If the auditor removed or heavily modified the text structure, flag as potential hallucination
            is_hallucination = False
            if len(initial_explanation.strip()) != len(final_audited_report.strip()):
                is_hallucination = True
                logger.warning(f"Hallucination mitigation triggered by Auditor for entry type: {disease.upper()}")

            logger.info("Shipping transactional entry metrics to PostgreSQL database...")
            log_evaluation(
                category=disease,
                score=risk_score,
                metrics=metrics,
                chunks=context_chunks,
                explanation=initial_explanation,
                report=final_audited_report,
                hallucination=is_hallucination
            )
            logger.info("PostgreSQL metrics transaction committed successfully.")
            
            return final_audited_report

        except Exception as e:
            logger.error(f"Execution boundary error encountered inside dual-LLM execution chain: {str(e)}")
            return f"❌ Error running dual-LLM pipeline: {str(e)}"
# Simulating main app context
if __name__ == "__main__":
    print("🚀 Testing MedicalRAGEngine with mock RAG inputs...")
    
    # Fake data simulating your ML models and future Postgres database
    mock_disease = "Stroke"
    mock_risk = 0.842
    mock_metrics = {"age": 67, "hypertension": 1}
    mock_db_chunks = [
        "AHA Guidelines: High glucose and hypertension compound stroke hazards.",
        "Protocol: For calculated risk exceeding 80%, prioritize immediate BP control."
    ]
    
    # Run the engine using the exported key
    engine = MedicalRAGEngine()
    report_text = engine.generate_clinical_explanation(
        disease=mock_disease,
        risk_score=mock_risk,
        metrics=mock_metrics,
        context_chunks=mock_db_chunks
    )
    print("\n--- LLM GENERATED REPORT OUTPUT ---")
    print(report_text)    