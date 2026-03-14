# DataAgent: Autonomous SQL Execution for BigQuery

DataAgent is an autonomous SQL generation and execution system that bridges the gap between natural language and structured BigQuery data. Designed for enterprise-grade efficiency, the system leverages a high-performance 8B parameter LLM architecture to translate user queries into valid SQL, execute them against live infrastructure, and synthesize results with high analytical accuracy.



## Agent Architecture & Data Pipeline
DataAgent utilizes a state-machine architecture to ensure that each stage of the data lifecycle—from query planning to result synthesis—is isolated and verifiable.

1. **Planning Node:** Analyzes user intent, identifies necessary schema relationships, and outlines logical steps before code generation.
2. **Generation Node:** Translates intent into optimized BigQuery SQL using an 8B parameter model, constrained by custom "Logical Safeguards."
3. **Execution Node:** Executes generated SQL against production tables (`fact_trips`) with built-in error handling and schema validation.
4. **Reporting Node:** Synthesizes normalized database tuples into human-readable narratives, grounded strictly within the metadata context.

---

## 🚀 Technical Challenges & Solutions

### 1. Cloud Authentication & Security
* **Challenge:** Modern Google Cloud SDKs are highly sensitive and automatically attempt to query the **Google Compute Engine (GCE) metadata service** for credentials. When running on non-GCE environments (like Streamlit Cloud), this results in `ConnectTimeout` errors.
* **Solution:** Implemented a global security override and a custom **credential-bridge** using `tempfile`. This forces the application to bypass network discovery and use explicit, secure service account credentials provided via memory-safe secret injection.

### 2. Agentic SQL Orchestration
* **Challenge:** Translating natural language into SQL is prone to "logical hallucinations," where the model attempts to compare disparate data types (e.g., total trip counts vs. average durations) or filter by single-day intervals that lack statistical significance.
* **Solution:** Built a stateful agent that enforces a "Logical Safeguards" layer. I integrated strict system prompt instructions that mandate unit-consistency checks, windowed date filtering (30-day aggregates), and defensive casting for accurate percentage calculations.



### 3. Cross-Environment Parity
* **Challenge:** Managing a development-to-production workflow is difficult when handling sensitive service account JSON files. Hard-coding paths causes deployment failures, and pushing secrets to GitHub is a security risk.
* **Solution:** Architected a hybrid authentication logic that detects the execution environment. The application safely switches between local JSON file sourcing (protected by `.gitignore`) and cloud-based secret management, ensuring a reproducible deployment without exposing credentials.

### 4. High-Fidelity BigQuery Integration
* **Challenge:** BigQuery is a dialect-specific, strict engine. Generating prompt-compliant SQL that adheres to project-qualified table names and strict typing required significant prompt engineering.
* **Solution:** Developed a library of optimized SQL templates and refined the agent’s schema-mapping instructions. Through iterative testing, I achieved high-fidelity query execution that consistently translates natural language into statistically accurate insights across a dataset of **280k+ records**.

---

## Dataset & Technical Stack
* **Dataset:** Austin Bikeshare Public Data (`bi-project-489517.austin_bikeshare`)
* **LLM Engine:** Llama-3.1-8B-Instant (via Groq API)
* **Framework:** Streamlit, LangGraph, Google Cloud BigQuery API
* **Data Warehouse:** Google BigQuery
* **Agentic Framework:** LangGraph / LangChain
* **Deployment:** Streamlit Cloud
* **Data Formatting:** `tabulate` for Markdown-based context injection
