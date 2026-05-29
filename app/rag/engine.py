import os 
from dotenv import load_dotenv
import psycopg2 # Added to handle DB connection check
from groq import Groq 
from utils.logger import logger  # 👈 System text log stream
from utils.audit_logger import log_evaluation  # 👈 Postgres evaluation logger

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

class MedicalRAGEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            logger.critical("Engine initialization failed: Missing GROQ_API_KEY environment variable.")
            raise ValueError("❌ Missing Groq API Key. Please set it in your terminal environment.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant"
        logger.info(f"MedicalRAGEngine initialized successfully using core model: {self.model}")

    def generate_clinical_explanation(self, disease: str, risk_score: float, patient_metrics: dict, context_chunks: list, status_updater=None) -> str:
        formatted_context = "\n\n".join([f"- {chunk}" for chunk in context_chunks])
        
        system_prompt = (
            "You are an expert Clinical Systems Analyst and Medical AI.\n"
            "Your task is to write a cohesive, professional evaluation of a patient's lab metrics based ONLY on the provided reference texts.\n\n"
            "CRITICAL WRITING INSTRUCTIONS:\n"
            "- Do NOT create a robotic list of every single metric saying 'not mentioned'. If a metric is not mentioned in the text, do not list it as a separate bullet point.\n"
            "- Instead, focus heavily on the data that IS present in the reference text (e.g., Albuminuria, Hemoglobin, CKD progression, RASi treatment).\n"
            "- Synthesize the patient's overall status. For example, note how their Albumin level of 1 or Hemoglobin level relates to the broad CKD outcomes and severities discussed in the guidelines.\n"
            "- Write in a narrative, professional medical tone. Cite your sources using (SOURCE: CATEGORY | CHUNK: ID) naturally within sentences."
        )
                
        user_prompt = f"""
        ### INSTRUCTIONS
        Cross-reference the [PATIENT METRICS] with the [VERIFIED CLINICAL GUIDELINES]. 
        Explain how the patient's specific metrics correlate to the risk framework *strictly* using the provided text.

        ### RISK ASSESSMENT FRAMEWORK (CRITICAL EVALUATION SCALE)
        - Risk Score Range 0.00 to 0.30: LOW RISK. Focus on standard lifestyle preventative measures.
        - Risk Score Range 0.31 to 0.70: MODERATE RISK. Requires clinical review, metabolic monitoring, and primary prevention strategies.
        - Risk Score Range 0.71 to 1.00: HIGH RISK. Requires immediate clinical evaluation, aggressive risk factor management, and secondary preventative interventions.

        [PATIENT METRICS]
        - Targeted Condition: {disease.upper()}
        - Model Classification Score: {risk_score * 100:.1f}%
        - Raw Input Features: {patient_metrics}

        [VERIFIED CLINICAL GUIDELINES]
        {formatted_context if formatted_context else "CRITICAL ERROR: No clinical guidelines provided."}

        ### REQUIRED OUTPUT FORMAT
        Provide your analysis using the following layout structure. If a section cannot be completed using ONLY the guidelines, write "Data not present in reference text."

        #### 1. Risk Score Contextualization
        (Explain what the {risk_score * 100:.1f}% score means based *only* on thresholds found in the guidelines)

        #### 2. Metric Evaluation against Guidelines
        (Cross-reference the raw features {patient_metrics} with explicit rules in the guidelines)

        #### 3. Clinical Disclaimers / Missing Data
        (List any patient metrics that were completely missing or unaddressed by the retrieved guidelines)
        """

        try:
            # --- STAGE 1: LOG GENERATION INITIATION ---
            logger.info(f"Sending Stage 1 prompt to Groq for disease framework: {disease.upper()}")
            if status_updater:
                status_updater.update(label="🧠 Stage 1: Analyzing patient metrics against clinical guidelines...", state="running")

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
            if status_updater:
                status_updater.update(label="🛡️ Stage 2: Adversarial Auditor fact-checking for hallucinations...", state="running")

            guard_system_prompt = (
            "You are a adversarial Medical AI Fact-Checker. Your sole function is to catch hallucinations.\n\n"
            "Compare the [GENERATED ASSESSMENT] against the [RAW SOURCE MEDICAL TRUTH].\n"
            "Identify every medical claim, benchmark, drug, or numerical range mentioned in the assessment.\n\n"
            "CRITICAL TEST:\n"
            "Is that exact claim or benchmark supported word-for-word or conceptually by the [RAW SOURCE MEDICAL TRUTH]?\n"
            "- If YES: Retain the sentence completely.\n"
            "- If NO / partial match: You must completely delete the sentence or paragraph.\n\n"
            "Output ONLY the finalized, safely stripped version of the assessment. Do not add introductions or conclusions."
            )

            guard_user_prompt = f"""
                ### SYSTEM VERIFIED TRUTHS (DO NOT DELETE CLAIMS BASED ON THESE)

                1. RISK ASSESSMENT FRAMEWORK EVALUATION SCALE:
                - Risk Score Range 0.00 to 0.30: LOW RISK. Focus on standard lifestyle preventative measures.
                - Risk Score Range 0.31 to 0.70: MODERATE RISK. Requires clinical review, metabolic monitoring, and primary prevention strategies.
                - Risk Score Range 0.71 to 1.00: HIGH RISK. Requires immediate clinical evaluation, aggressive risk factor management, and secondary preventative interventions.

                ### RAW SOURCE MEDICAL TRUTH (REFERENCE ONLY)
                {formatted_context if formatted_context else "NO REFERENCE DATA AVAILABLE."}

                ### GENERATED ASSESSMENT TO AUDIT
                {initial_explanation}

                ### AUDIT EXECUTION DIRECTIVE
                Review every assertion, metric range, and recommendation in the [GENERATED ASSESSMENT]. 
                Cross-reference it with both the SYSTEM VERIFIED TRUTHS and the RAW SOURCE MEDICAL TRUTH. 

                - If a claim about a risk tier matches the 3-tier framework scale above, RETAIN IT.
                - If a claim about a lab metric matches the reference values or the PDF chunks, RETAIN IT.
                - If any clinical claim is completely missing from both references, delete that sentence or paragraph entirely.
                
                Output ONLY the finalized, safely stripped text. No explanations, no pleasantries.
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
            if status_updater:
                status_updater.update(label="💾 Stage 3: Committing telemetry metrics to PostgreSQL database...", state="running")
            
            is_hallucination = False
            if len(initial_explanation.strip()) != len(final_audited_report.strip()):
                is_hallucination = True
                logger.warning(f"Hallucination mitigation triggered by Auditor for entry type: {disease.upper()}")

            logger.info("Shipping transactional entry metrics to PostgreSQL database...")
            log_evaluation(
                category=disease,
                score=risk_score,
                metrics=patient_metrics,
                chunks=context_chunks,
                explanation=initial_explanation,
                report=final_audited_report,
                hallucination=is_hallucination
            )
            if status_updater:
                status_updater.update(label="✅ Clinical report fully audited and verified!", state="complete")

            logger.info("PostgreSQL metrics transaction committed successfully.")
            
            return final_audited_report

        except Exception as e:
            logger.error(f"Execution boundary error encountered inside dual-LLM execution chain: {str(e)}")
            if status_updater:
                status_updater.update(label="❌ Error running dual-LLM pipeline", state="error")
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
        patient_metrics=mock_metrics,
        context_chunks=mock_db_chunks
    )
    print("\n--- LLM GENERATED REPORT OUTPUT ---")
    print(report_text)    