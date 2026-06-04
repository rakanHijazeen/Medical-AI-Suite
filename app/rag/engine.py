import os 
import re
import json
import time
from typing import Optional
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

    def _stream_and_collect(self, call_fn, stream_callback=None, status_updater=None, label: str = "Streaming:"):
        """
        Helper to call model API with best-effort streaming. Accepts a zero-arg callable
        that invokes the underlying SDK call (preferably with streaming enabled).
        Calls `stream_callback` for each incremental chunk when available and
        returns the fully concatenated text.
        """
        collected = ""
        try:
            # Try streaming mode first (SDKs often accept `stream=True`)
            resp_iter = call_fn(stream=True)
            for chunk in resp_iter:
                try:
                    # Attempt common delta/message shapes
                    part = None
                    if hasattr(chunk, "choices"):
                        c0 = chunk.choices[0]
                        # delta content
                        if hasattr(c0, "delta") and getattr(c0.delta, "content", None):
                            part = c0.delta.content
                        # full message
                        elif hasattr(c0, "message") and getattr(c0.message, "content", None):
                            part = c0.message.content
                    # Fallback: if chunk is plain str
                    if part is None and isinstance(chunk, str):
                        part = chunk
                    if part:
                        collected += part
                        if stream_callback:
                            try:
                                stream_callback(part)
                            except Exception:
                                logger.debug("stream_callback raised an exception; continuing.")
                except Exception:
                    continue
            return collected
        except TypeError:
            # SDK may not accept stream param; fall back to non-streaming
            resp = call_fn()
            text = None
            try:
                text = resp.choices[0].message.content
            except Exception:
                # Last resort: try attr 'text'
                text = getattr(resp, 'text', str(resp))
            if stream_callback and text:
                self._stream_text(text, status_updater=status_updater, stream_callback=stream_callback, label=label)
            return text or ""

    def _stream_text(self, text: str, status_updater=None, stream_callback=None, label: str = "Streaming:"):
        """Chunk a full text and push incremental updates to status_updater or stream_callback.

        Used when SDK streaming is not available but the UI still expects
        incremental updates.
        """
        if not text:
            return
        chunk_size = 250
        for i in range(0, len(text), chunk_size):
            piece = text[i:i+chunk_size]
            if stream_callback:
                try:
                    stream_callback(piece)
                except Exception:
                    logger.debug("stream_callback raised an exception during fallback streaming.")
            if status_updater:
                try:
                    status_updater.update(label=f"{label} {piece}", state="running")
                except Exception:
                    pass
            time.sleep(0.02)

    def _parse_citations_from_stage1(self, stage1_output: str) -> list:
        """Extract citation IDs like [CHUNK: 2] or JSON {"citations": [0,1]} from the LLM output."""
        if not stage1_output:
            return []
        citations = []
        # Find bracket citations [CHUNK: 2], [GUIDELINE_ID: A4]
        bracket_pattern = r"\[(?:CHUNK|GUIDELINE_ID|SOURCE|ID):\s*([A-Za-z0-9_\-]+)\]"
        for m in re.findall(bracket_pattern, stage1_output):
            citations.append(m)

        # Try to parse JSON-style citations if present
        try:
            # look for a JSON object with a citations key
            json_match = re.search(r"\{.*?\"citations\"\s*:\s*\[.*?\].*?\}", stage1_output, re.DOTALL)
            if json_match:
                j = json.loads(json_match.group(0))
                if isinstance(j.get("citations"), list):
                    citations.extend([str(x) for x in j.get("citations")])
        except Exception:
            logger.debug("No JSON citations parsed from Stage 1 output.")

        # Deduplicate preserving order
        seen = set()
        out = []
        for c in citations:
            if c not in seen:
                seen.add(c)
                out.append(c)
        logger.info(f"Extracted citations from Stage1: {out}")
        return out

    def _filter_context_by_citations(self, context_chunks: list, citations: list) -> list:
        """Return only chunks referenced by the citations list. Citations may be indices or substrings."""
        if not citations:
            logger.warning("No citations extracted; returning full context_chunks.")
            return context_chunks
        filtered = []
        for c in citations:
            # try index
            try:
                idx = int(c)
                if 0 <= idx < len(context_chunks):
                    filtered.append(context_chunks[idx])
                    continue
            except Exception:
                pass
            # match by substring or chunk id prefix
            for chunk in context_chunks:
                if c.lower() in chunk.lower() or chunk.startswith(f"- {c}") or chunk.startswith(c):
                    if chunk not in filtered:
                        filtered.append(chunk)
                    break
        if not filtered:
            logger.warning("Filtering produced empty set; returning full context_chunks.")
            return context_chunks
        logger.info(f"Context filtered to {len(filtered)} chunks from {len(context_chunks)} total.")
        return filtered

    def _format_chunks_with_indices(self, context_chunks: list) -> str:
        if not context_chunks:
            return "NO CONTEXT CHUNKS PROVIDED."
        formatted = []
        for i, chunk in enumerate(context_chunks):
            formatted.append(f"[CHUNK: {i}] {chunk}")
        return "\n\n".join(formatted)

    def generate_clinical_explanation(self, disease: str, risk_score: float, patient_metrics: dict, context_chunks: list, status_updater=None, stream_callback=None) -> str:
        """
        Generate a clinical explanation and run a two-stage audit while streaming
        partial outputs to `stream_callback` when available.

        Args:
            disease: disease name
            risk_score: model probability (0-1)
            patient_metrics: input metrics dict
            context_chunks: list of guideline text chunks
            status_updater: optional status UI object
            stream_callback: optional callable(str) to receive incremental text

        Returns:
            Final audited report string
        """
        system_prompt = (
            "You are an expert Clinical Systems Analyst and Medical AI.\n"
            "Your task is to write a cohesive, professional evaluation of a patient's lab metrics based ONLY on the provided reference texts.\n\n"
            "CITATION REQUIREMENT:\n"
            "- For every clinical guideline, reference range, or recommendation you cite, include the specific source chunk ID in [CHUNK: ID] format.\n"
            "- Example: 'According to the CKD progression framework [CHUNK: 3], hemoglobin below 12 g/dL indicates anemia complications.'\n\n"
            "CRITICAL WRITING INSTRUCTIONS:\n"
            "- Do NOT create a robotic list of every single metric saying 'not mentioned'. If a metric is not mentioned in the text, do not list it as a separate bullet point.\n"
            "- Instead, focus heavily on the data that IS present in the reference text (e.g., Albuminuria, Hemoglobin, CKD progression, RASi treatment).\n"
            "- Synthesize the patient's overall status. For example, note how their Albumin level of 1 or Hemoglobin level relates to the broad CKD outcomes and severities discussed in the guidelines.\n"
            "- Write in a narrative, professional medical tone for a peer-reviewed clinical audience. Avoid jargon that obscures the clinical reasoning. Ensure every external claim is tagged with its source chunk."
            "- If you cannot find a supporting chunk for a claim, do not make a claim. Do not invent [CHUNK: X] tags. If the reference is not in the text, omit the claim entirely."
        )

        user_prompt = f"""
        ### INSTRUCTIONS
        Cross-reference the [PATIENT METRICS] with the [VERIFIED CLINICAL GUIDELINES]. 
        Explain how the patient's specific metrics correlate to the risk framework *strictly* using the provided text.
        
        IMPORTANT: After each guideline reference or clinical claim, include [CHUNK: X] tags to indicate which source chunk(s) you used.

        ### RISK ASSESSMENT FRAMEWORK (CRITICAL EVALUATION SCALE)
        - Risk Score Range 0.00 to 0.30: LOW RISK. Focus on standard lifestyle preventative measures.
        - Risk Score Range 0.31 to 0.70: MODERATE RISK. Requires clinical review, metabolic monitoring, and primary prevention strategies.
        - Risk Score Range 0.71 to 1.00: HIGH RISK. Requires immediate clinical evaluation, aggressive risk factor management, and secondary preventative interventions.

        [PATIENT METRICS]
        - Targeted Condition: {disease.upper()}
        - Model Classification Score: {risk_score * 100:.1f}%
        - Raw Input Features: {patient_metrics}

        [VERIFIED CLINICAL GUIDELINES]
        {self._format_chunks_with_indices(context_chunks)}

        ### REQUIRED OUTPUT FORMAT
        Provide your analysis using the following layout structure. If a section cannot be completed using ONLY the guidelines, write "Data not present in reference text."
        Include [CHUNK: X] citations for every external reference.

        #### 1. Risk Score Contextualization
        (Explain what the {risk_score * 100:.1f}% score means based *only* on thresholds found in the guidelines [CHUNK: X])

        #### 2. Metric Evaluation against Guidelines
        (Cross-reference the raw features {patient_metrics} with explicit rules in the guidelines [CHUNK: X, Y, Z]), if any specific metric does not directly correlate with any specific guideline recommendations in the provided text DO NOT mention it at all in the output. Focus only on synthesizing the information that is present in the guidelines with the patient's metrics to provide a cohesive clinical picture.)


        #### 3. Clinical Disclaimers / Missing Data
        (Only list any critical clinical information that is missing from the patient's metrics but is mentioned in the guidelines as important for risk assessment). 
        """

        try:
            # --- STAGE 1: GENERATE WITH CITATIONS (STREAMING WHEN AVAILABLE) ---
            logger.info(f"Stage 1: Analyzing patient metrics for disease: {disease.upper()}")
            if status_updater:
                status_updater.update(label="🧠 Stage 1: Analyzing patient metrics against clinical guidelines...", state="running")

            def _call_stage1(**kwargs):
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    temperature=0.2,
                    max_tokens=1024,
                    seed=42,
                    **kwargs
                )

            # Prefer the provided stream_callback; otherwise stream to status_updater
            if stream_callback:
                initial_explanation = self._stream_and_collect(
                    _call_stage1,
                    stream_callback=stream_callback,
                    status_updater=status_updater,
                    label="🧠 Stage 1 (stream):"
                )
            else:
                def _stage1_cb(part: str):
                    if status_updater:
                        try:
                            status_updater.update(label=f"🧠 Stage 1 (stream): {part}", state="running")
                        except Exception:
                            pass
                initial_explanation = self._stream_and_collect(
                    _call_stage1,
                    stream_callback=_stage1_cb,
                    status_updater=status_updater,
                    label="🧠 Stage 1 (stream):"
                )
            logger.info("Stage 1: Clinical explanation generated with citations.")

            # --- EXTRACT CITATIONS & FILTER CONTEXT ---
            logger.info("Extracting citations from Stage 1 output...")
            citations = self._parse_citations_from_stage1(initial_explanation or "")
            logger.info("Filtering context chunks based on Stage 1 citations...")
            filtered_context_chunks = self._filter_context_by_citations(context_chunks, citations)
            # Sort the filtered list alphabetically before building the string
            filtered_context_chunks.sort()
            filtered_context = "\n\n".join([f"- {chunk}" for chunk in filtered_context_chunks])

            # =====================================================================
            # STAGE 2: AUDITOR WITH FILTERED CONTEXT ONLY (STREAMING WHEN AVAILABLE)
            # =====================================================================
            logger.info("Stage 2: Passing auditor with filtered context for efficiency...")
            if status_updater:
                status_updater.update(label="🛡️ Stage 2: Adversarial Auditor fact-checking for hallucinations...", state="running")

            guard_system_prompt = (
                "You are an adversarial Medical AI Fact-Checker and Editor. Your sole function is to catch and eliminate hallucinations while preserving structural integrity.\n\n"
                "Compare the [GENERATED ASSESSMENT] against the [FILTERED SOURCE MEDICAL TRUTH].\n"
                "Identify every medical claim, benchmark, metric range, or clinical guideline mentioned.\n\n"
                "CRITICAL TEST:\n"
                "Is the specific clinical claim or numeric benchmark supported conceptually or word-for-word by either the SYSTEM VERIFIED TRUTHS or the FILTERED SOURCE MEDICAL TRUTH?\n"
                "- If YES: Retain the sentence exactly as written.\n"
                "- If NO / UNVERIFIED: You must rewrite or edit the sentence to completely remove the unverified claim, or delete only the non-compliant sentence.\n\n"
                "CRITICAL STRUCTURAL RULES:\n"
                "1. Preserve all Markdown formatting elements (e.g., headers like '###', bullet points, bold text) exactly as they appear in the input. Do not leave trailing or empty markdown bullets.\n"
                "2. Do not delete an entire paragraph if only one sentence contains an unverified claim; isolate and remove or correct only the offending sentence.\n"
                "3. DO NOT COPY THE PROMPT HEADINGS. DO NOT echo back the 'SYSTEM VERIFIED TRUTHS' or 'FILTERED SOURCE MEDICAL TRUTH' sections.\n"
                "4. OUTPUT ONLY the cleaned version of the text found under '### GENERATED ASSESSMENT TO AUDIT'. Absolutely no intro, no metadata, no summary labels."
            )            

            guard_user_prompt = f"""
                ### INSTRUCTIONS
                Review every assertion, metric range, and recommendation in the [GENERATED ASSESSMENT]. 
                Cross-reference it with both the SYSTEM VERIFIED TRUTHS and the FILTERED SOURCE MEDICAL TRUTH. 

                - If a claim about a risk tier matches the 3-tier framework scale above, RETAIN IT.
                - If a claim about a lab metric matches the filtered reference values, RETAIN IT.
                - If a sentence makes an unverified specific clinical claim (e.g., an unreferenced diagnostic cutoff or a specific drug prescription) that appears in neither source, STRIP OR REPHRASE that specific sentence to render it safe and generalized.
                - Maintain the original document's markdown layout structure perfectly.
                
                Output ONLY the finalized, safely stripped text. No explanations, no pleasantries.

                ### SYSTEM VERIFIED TRUTHS (DO NOT DELETE CLAIMS BASED ON THESE)
                1. RISK ASSESSMENT FRAMEWORK EVALUATION SCALE:
                - Risk Score Range 0.00 to 0.30: LOW RISK. Focus on standard lifestyle preventative measures.
                - Risk Score Range 0.31 to 0.70: MODERATE RISK. Requires clinical review, metabolic monitoring, and primary prevention strategies.
                - Risk Score Range 0.71 to 1.00: HIGH RISK. Requires immediate clinical evaluation, aggressive risk factor management, and secondary preventative interventions.

                ### FILTERED SOURCE MEDICAL TRUTH (STAGE 1 REFERENCED CHUNKS ONLY)
                {filtered_context if filtered_context.strip() else "NO REFERENCED CHUNKS AVAILABLE."}

                ### GENERATED ASSESSMENT TO AUDIT
                {initial_explanation}
                """

            def _call_stage2(**kwargs):
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": guard_system_prompt}, {"role": "user", "content": guard_user_prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                    seed=42,
                    **kwargs
                )

            if stream_callback:
                final_audited_report = self._stream_and_collect(
                    _call_stage2,
                    stream_callback=stream_callback,
                    status_updater=status_updater,
                    label="🛡️ Stage 2 (stream):"
                )
            else:
                def _stage2_cb(part: str):
                    if status_updater:
                        try:
                            status_updater.update(label=f"🛡️ Stage 2 (stream): {part}", state="running")
                        except Exception:
                            pass
                final_audited_report = self._stream_and_collect(
                    _call_stage2,
                    stream_callback=_stage2_cb,
                    status_updater=status_updater,
                    label="🛡️ Stage 2 (stream):"
                )
            logger.info("Stage 2: Clinical audit evaluation complete.")

            # --- STAGE 3: TELEMETRY EVALUATION LOGGING ---
            if status_updater:
                status_updater.update(label="💾 Stage 3: Committing telemetry metrics to PostgreSQL database...", state="running")

            is_hallucination = False
            if (initial_explanation or "").strip() != (final_audited_report or "").strip():
                is_hallucination = True
                logger.warning(f"Hallucination mitigation triggered by Auditor for entry type: {disease.upper()}")

            logger.info(f"Token efficiency: Original context size={len(context_chunks)}, Filtered size={len(filtered_context_chunks)}")
            logger.info("Shipping transactional entry metrics to PostgreSQL database...")
            log_evaluation(
                category=disease,
                score=risk_score,
                metrics=patient_metrics,
                chunks=filtered_context_chunks,  # Log only the filtered chunks used
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