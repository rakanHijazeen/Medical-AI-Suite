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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # Low temperature forces the model to stay factual
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

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