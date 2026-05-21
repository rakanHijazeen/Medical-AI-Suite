import os 
from groq import Groq 

class MedicalRAGEngine:
    def __init__(self, api_key: str = None):
        # Securely grab the key from the terminal environment
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ Missing Groq API Key. Please set it in your terminal environment.")
        
        # Initialize the official Groq client
        self.client = Groq(api_key=self.api_key)
        
        # Llama 3 8B has incredibly fast processing speeds
        self.model = "llama-3.1-8b-instant"

    def generate_clinical_explanation(self, disease: str, risk_score: float, metrics: dict, context_chunks: list) -> str:
        # 1. Take the list of database strings and join them into one clean text block
        formatted_context = "\n\n".join([f"- {chunk}" for chunk in context_chunks])
        
        # 2. The System Prompt acts as the "Judge" rulebook to keep the AI accurate
        system_prompt = (
            "You are an expert clinical AI assistant. Rely ONLY on the provided "
            "Medical Reference Guidelines context. Do not assume or extrapolate facts. "
            "Do not include conversational fluff. Use clean Markdown formatting."
        )
        
        # 3. The User Prompt injects the live data from your Streamlit inputs
        user_prompt = f"""
### PATIENT CLINICAL DATA
- **Target Disease Assessment:** {disease.upper()}
- **Calculated Risk Score:** {risk_score * 100:.1f}%
- **Patient Metrics Evaluated:** {metrics}

### MEDICAL REFERENCE GUIDELINES (Retrieved Context)
{formatted_context if formatted_context else "No reference guidelines found."}
"""

        # 4. Fire the structured request to Groq's cloud servers
        try:
            gen_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # Low temperature forces the model to stay factual
                max_tokens=1024
            )
            initial_explanation = gen_response.choices[0].message.content
            # =====================================================================
            # STAGE 2: THE GUARD / AUDITOR LLM
            # =====================================================================
            print("🛡️ [RAG GUARD] Passing generation to Auditor LLM for medical compliance verification...")
            
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

            # 2. Fire the audit verification request
            audit_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": guard_system_prompt},
                    {"role": "user", "content": guard_user_prompt}
                ],
                temperature=0.1, # Extremely low temperature for strict adherence to text
                max_tokens=1024
            )
            
            final_audited_report = audit_response.choices[0].message.content
            return final_audited_report

        except Exception as e:
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