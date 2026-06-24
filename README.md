![Medical AI Suite Demo](assets/demo.gif)

# Medical AI Suite 🩺

An intelligent clinical decision support platform that combines multi-disease predictive machine learning classifiers with a hallucination-resistant, guideline-anchored Retrieval-Augmented Generation (RAG) auditing pipeline.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Machine Learning & RAG Pipeline](#-machine-learning--rag-pipeline)
- [Challenges & Technical Hurdles](#-challenges--technical-hurdles)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#️-installation--setup)
- [Usage](#️-usage)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

## 📌 Project Overview

The **Medical AI Suite** is an advanced Clinical Decision Support (CDS) platform designed for healthcare professionals to safely bridge the gap between quantitative machine learning risk metrics and qualitative medical guidelines.

While standard ML models excel at calculating risk percentages from raw clinical biomarkers, these probabilistic outputs often exist as an interpretive "black box" for front-line clinicians. This platform directly addresses that problem by feeding the calibrated risk outputs into a localized, containerized vector database, generating a dual-stage, adversarial-audited narrative report fully anchored in international medical standards.

---

## 🏗 System Architecture

[User Input: Lab Values] ──> [StandardScaler Preprocessing] ──> [ML Predictive Layer (RF / LogReg)] ──> [Risk Probabilities]
│
▼
<── [HNSW Context Retrieval]
[Structured Audit Report] <── [Stage 2: Adversarial Auditor] <── [Stage 1: LLM Generator]

1. **Input:** The clinician provides multi-dimensional patient parameters via a unified web interface.
2. **ML Inference:** Tabular classifiers process inputs to output calibrated risk probabilities.
3. **Knowledge Retrieval:** Dense vector representations isolate corresponding medical context via local HNSW indexing.
4. **Adversarial Synthesis:** A dual-stage LLM layout drafts and aggressively filters diagnostic narratives to enforce strict factual compliance.

---

## 🚀 Features

### Phase 1: Automated Clinical Auditing (Fully Functional)

- **Multi-Condition Risk Prediction:** Instant classification targeting Kidney Disease, Heart Disease, Diabetes, and Stroke.
- **Calibrated ML Architectures:** Tailored predictive modeling using Random Forests and Logistic Regression.
- **Guideline-Anchored RAG Integration:** Automatic embedding retrieval mapping patient metrics straight to official clinical protocols (ADA, AHA, HFSA, KDIGO).
- **Dual-Stage Adversarial Auditing:** A two-tier LLM configuration (`llama-3.1-8b-instant` via Groq) where an independent Auditor cross-checks generated text sentence-by-sentence to structurally eliminate hallucinations.
- **Dynamic Interface Logging:** A Streamlit frontend tracking background execution milestones live using a clean generation expander.
- **Professional PDF Report Generation:** Instantly exports comprehensive patient records containing vital charts, target values, feature impacts, and the finalized clinical audit statement.

### 💬 Phase 2: Conversational Extension (Fully Functional Deployment)

The system features a fully deployed, high-specificity conversational layer built directly into the workspace tabs. This extension shifts the platform from a static evaluation summary tool into an active, multi-turn clinical reasoning partner.

#### Key Functional Capabilities:

- **State-Managed Interactive Layer:** Built on a persistent, multi-turn session tracking lifecycle (`st.session_state`). Clinicians can cross-examine generated reports, probe risk factors, or follow up on specific treatment pathways across multiple messages without losing historical context.
- **Hallucination-Resistant Bounding:** The underlying chat agent (`MedicalChatEngine`) operates within strict retrieval-augmented constraints. It hooks directly into the localized vector database, ensuring answers are strictly grounded in validated medical standards (e.g., KDIGO, AHA/ASA guidelines) and any attached context documents, heavily mitigating large language model hallucinations.
- **Dynamic Context Injection:** Leverages a seamlessly integrated file uploader system. Clinicians can drag and drop external patient histories, laboratory spreadsheets, or clinical notes (such as PDFs) directly into the stream, automatically expanding the context window for the active evaluation session.
- **Tailored CSS UI/UX Architecture:** The interface is wrapped in a custom corporate stylesheet (`style_chat.css`) utilizing precise CSS injection overrides (`!important`). This ensures the theme maintains complete dark-slate continuity (`#030712`, `#0f1626`) across complex, dynamic Streamlit components—such as multi-file dropzone status cards, inner text block markdown strings, contextual boundaries, and independent sidebar layouts (`streamlit_option_menu`).

### 📋 Clinical Guidelines & Verification Repositories (Evidence Base)

To guarantee full transparency and absolute clinical traceability, the application includes a dedicated **Clinical Guidelines Workspace**. Rather than keeping the underlying medical logic hidden behind the conversational interface, this workspace maps out and displays the authoritative foundations directly to the user.

#### Features & Structure:

- **Authoritative Reference Matrix:** Organizes clinical knowledge bases into responsive, high-contrast visual summary cards grouped by medical discipline (e.g., _KDIGO Guidelines_ for renal filtration thresholds/SGLT2i safety guardrails, and _AHA/ASA Consensus Protocols_ for stroke biomarker tracking).
- **Direct Proof/Source Linking:** Each guideline element is tightly coupled with its official, peer-reviewed source artifact. Clinicians can instantly launch the external reference URLs to perform a quick manual validation against raw literature definitions.
- **Biomarker Range Reference Tool:** Serves as a transparent clinical baseline ledger, making it simple to inspect target value metrics and confirm that the automated audit summaries align perfectly with the newest published standard practices.

---

## 🧠 Machine Learning & RAG Pipeline

### 1. Tabular Data Preprocessing & Training

- **Data Cleansing:** Missing or null variables (predominant in raw stroke and kidney records) are handled via localized mode/mean imputation and structured field filtering.
- **Feature Engineering & Normalization:** Continuous metrics are scaled using `StandardScaler` to ensure uniform variance boundaries across linear and tree-based setups.
- **Imbalance Handling:** Synthetic Minority Over-sampling Technique (**SMOTE**) is applied during development phases to address underlying target class distribution skewness.
- **Validation Constraints:** Pipelines are split utilizing `stratify=y` to lock matching target densities across development samples.
- **Model Selection Matrix:**
  - **Kidney Disease & Diabetes:** Deployed using optimized **Random Forest Classifiers**.
  - **Heart Disease & Stroke:** Deployed using optimized **Logistic Regression Models**.

### 2. Knowledge Base Retrieval (RAG Strategy)

- **Ingested Source Literature:** Complete textual datasets parsed from official medical protocols:
  - _ADA Diagnosis and Classification of Diabetes: Standards of Care (2026)_
  - _AHA/ACC/HFSA Guideline for the Management of Heart Failure_
  - _KDIGO Clinical Practice Guideline for Diabetes and CKD (2026)_
  - _AHA/ASA Guideline for the Early Management of Acute Ischemic Stroke (2026)_
- **Vector Pipeline:** Content is token-split with a strict 10% semantic overlapping safety boundary, transformed via a dense bi-encoder text embedding model (`all-MiniLM-L6-v2`), and indexed using a **Hierarchical Navigable Small World (HNSW)** graph structure inside a database vector ecosystem.

---

## 🚀 Challenges & Technical Hurdles

### 1. 🚨 Model Logic Inversion & Scaling Mismatch

**Problem:** Initial testing of the Heart Disease model yielded "100% Healthy" predictions even for high-risk inputs (e.g., 80-year-old with high BP). This was traced to a mismatch between the training vector (23 features) and the application input (5 features).
**🔧 Action:** Re-engineered the data pipeline to synchronize the `StandardScaler` with a new 6-feature numerical set (including a custom interaction term) and aligned the one-hot encoded categorical indices.
**✅ Result:** Restored model sensitivity, achieving a realistic 97% risk probability for critical cases.

### 2. 🚨 Overfitting & "Perfect" Metric Bias (Kidney Model)

**Problem:** The Chronic Kidney Disease (CKD) model achieved 100% accuracy/recall, indicating structural overfitting on highly correlated biomarkers like Hemoglobin and PCV.
**🔧 Action:** Implemented Gaussian Noise Injection to simulate clinical measurement variance and applied Structural Regularization by capping the Random Forest `max_depth` and increasing `min_samples_leaf`.
**✅ Result:** Traded artificial precision for clinical reliability, bringing metrics to a robust and generalizable 90-95%.

### 3. 🚨 Stroke Diagnostic Model Optimization

**Problem:** Initial model (XGBoost) produced too many false negatives (high risk) and excessive "false alarms" (clinical noise), undermining user trust.
**🔧 Action:** Swapped XGBoost for Logistic Regression to improve interpretability and tuned the decision threshold to 0.55 to prioritize high-risk detection.
**✅ Result:** Secured 80% Recall (minimizing missed cases) while slashing false alarms by 14% through improved precision.

### 4. 🚨 Explainable AI (XAI) Synchronization

**Problem:** Standard feature importance plots were static and did not reflect the specific reasoning for individual patient predictions. Furthermore, SHAP explainer states were "freezing" in the Streamlit UI.
**🔧 Action:** Developed a unified interpretability layer using SHAP (Shapley Additive Explanations) for non-linear models and coefficient-mapping for linear models. Fixed Matplotlib state issues using global figure clearing to ensure real-time visual updates.
**✅ Result:** A fully transparent diagnostic suite where every prediction is accompanied by a mathematically sound "local" explanation.

---

## 🛠 Tech Stack

- **Interface Frontend:** `Streamlit` (Includes dynamic pipeline execution tracking modules)
- **Predictive Core:** `scikit-learn`, `imbalanced-learn`
- **Embedding & Search Layer:** HuggingFace `all-MiniLM-L6-v2`, **PostgreSQL** with `pgvector` index integration (HNSW optimization graph)
- **Generative Infrastructure:** **Groq LPU Inference Engine** executing `llama-3.1-8b-instant`
- **Container Isolation & Deployment:** **Docker** / `docker-compose` ecosystem hosting local database structures
- **Data Interpretation & Graphics:** `matplotlib`, `seaborn`
- **Report Generation Engine:** `fpdf2`
- **Serialization Utilities:** `joblib`

---

## 📁 Repository Structure

- `app/` – Streamlit application code, page layouts, UI views, and custom dictionary mappings
- `data/` – De-identified sample datasets used for model evaluation and cleaning
- `models/` – Pre-trained production-ready serialized model and scaling artifacts
- `notebooks/` – Historical exploratory data analysis, SMOTE testing, and model architecture training tracks
- `knowledge_base/` - ingestion & embeding.
- `utils/` - logging scripts.

---

## ▶️ Installation & Setup

This project uses **Docker and Dev Containers** for a fully containerized development environment. We recommend the VS Code Dev Container approach for the smoothest setup experience.

### Option 1: VS Code Dev Container (⭐ Recommended)

The easiest way to get started with full IDE integration, debugging, and all dependencies pre-configured.

#### Prerequisites

- **Docker Desktop** (or Docker Engine on Linux)
  - [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
- **VS Code** with Remote - Containers extension
  - [Install Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- **Groq API Key**
  - Get it from [https://console.groq.com/](https://console.groq.com/)

#### Quick Start (3 Steps)

1. **Clone and navigate to the project:**

   ```bash
   git clone https://github.com/rakanHijazeen/Medical-AI-Suite.git
   cd Medical-AI-Suite
   ```

2. **Create environment configuration:**

   ```bash
   cp .devcontainer/.env.example .env
   ```

   Edit `.env` and add your credentials:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   DB_HOST=postgres
   DB_PORT=5432
   DB_NAME=medical_ai
   DB_USER=postgres
   DB_PASSWORD=postgres_dev_password
   ```

3. **Open in VS Code and reopen in container:**

   ```bash
   code .
   ```

   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Search for: **Remote-Containers: Reopen in Container**
   - Wait for container build (5-10 minutes on first run)

#### What Gets Installed

✅ Python 3.11 runtime  
✅ All dependencies from `requirements.txt`  
✅ PostgreSQL 16 database  
✅ Streamlit (port 8501)  
✅ Jupyter Lab (port 8888)  
✅ Development tools (pytest, black, isort, pylint, mypy)  
✅ 12+ VS Code extensions  
✅ Pre-configured debugging

#### Running Services

Once in the dev container:

```bash
# Start Streamlit app
make streamlit
# Access: http://localhost:8501

# Start Jupyter Lab
make jupyter
# Access: http://localhost:8888

# Run tests
make test

# Format code
make format

# View all commands
make help
```

For detailed instructions, see [`.devcontainer/README.md`](.devcontainer/README.md)

---

### Option 2: Local Setup (Manual)

If you prefer to set up locally without containers:

#### Prerequisites

- **Python 3.11+**
- **PostgreSQL 12+** installed and running locally
- **Groq API Key**

#### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/rakanHijazeen/Medical-AI-Suite.git
   cd Medical-AI-Suite
   ```

2. **Create Python virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**

   ```bash
   cp .devcontainer/.env.example .env
   ```

   Update `.env` with your PostgreSQL credentials and Groq API key:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=medical_ai
   DB_USER=your_postgres_user
   DB_PASSWORD=your_postgres_password
   ```

5. **Initialize database:**

   ```bash
   psql -U postgres -d medical_ai -f .devcontainer/init-db.sql
   ```

6. **Populate the RAG Knowledge Base:**
   ```bash
   python knowledge_base/init_db.py
   ```

#### Running the Application

```bash
# Start Streamlit app
streamlit run app/main.py

# The app will open at http://localhost:8501
```

---

### Environment Variables Reference

| Variable       | Required | Default      | Purpose                        |
| -------------- | -------- | ------------ | ------------------------------ |
| `GROQ_API_KEY` | ✅ Yes   | -            | Groq API key for LLM inference |
| `DB_HOST`      | ✅ Yes   | `postgres`   | Database hostname              |
| `DB_PORT`      | ✅ Yes   | `5432`       | Database port                  |
| `DB_NAME`      | ✅ Yes   | `medical_ai` | Database name                  |
| `DB_USER`      | ✅ Yes   | `postgres`   | Database user                  |
| `DB_PASSWORD`  | ✅ Yes   | -            | Database password              |
| `LOG_LEVEL`    | ❌ No    | `INFO`       | Logging verbosity              |

See `.devcontainer/.env.example` for all available options.

---

### Troubleshooting

**Dev Container won't build?**

```bash
# Rebuild from scratch
docker system prune -a
# Then: Remote-Containers: Rebuild Container in VS Code
```

**PostgreSQL connection fails?**

```bash
# Check container logs
docker logs medical_ai_postgres

# Or reset and restart
docker-compose -f .devcontainer/docker-compose.yml down -v
docker-compose -f .devcontainer/docker-compose.yml up -d
```

**Port already in use?**
Edit `docker-compose.yml` or `.env` to use different ports. See [`.devcontainer/README.md`](.devcontainer/README.md) for details.

For more help, see:

- [`.devcontainer/README.md`](.devcontainer/README.md) - Complete dev container guide
- [`.devcontainer/ARCHITECTURE.md`](.devcontainer/ARCHITECTURE.md) - Technical details
- [`.devcontainer/SETUP_SUMMARY.md`](.devcontainer/SETUP_SUMMARY.md) - Quick reference

  ▶️ Usage
  Start the platform web-app interface:

  ```bash
  streamlit run app/main.py
  ```

Select an assessment workspace via the navigation layout (Kidney, Heart, Diabetes, or Stroke).

Populate patient metrics and baseline laboratory values.

Expand the processing terminal drawer to monitor runtime pipeline execution milestones.

Review generated risk predictions alongside the dual-stage adversarial audited guidelines summary.

Export the completed analysis profile to an un-editable, formal PDF medical evaluation record.

🛡 Disclaimer
This project is intended strictly for educational, research, and technical evaluation purposes. It is not designed or certified to replace expert medical consultation, clinical diagnosis, or therapeutic choices. Always pursue guidance from a licensed health professional regarding personal clinical situations. Machine learning metrics output historical probability estimates and are bounded entirely by training data characteristics.

👤 Author
Rakan Hijazeen

GitHub: github.com/rakanHijazeen

LinkedIn: linkedin.com/in/rakan-hijazeen-327647392
