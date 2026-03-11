# DataAgent: Autonomous SQL Execution for BigQuery
[Click Here To Test The Agent](https://database-assistant-3brgrkrmo3vkpwtsqfr87r.streamlit.app)

DataAgent is an autonomous SQL generation and execution system that bridges the gap between natural language and structured BigQuery data. Designed for extreme efficiency, the system leverages a compact, high-performance LLM architecture to translate user queries into valid SQL, execute them, and interpret the results with high analytical accuracy.


## Agent Architecture & Data Pipeline

The DataAgent is built on a modular state-machine architecture, ensuring that each step of the query process is isolated, testable, and robust.

### The Pipeline Workflow
1.  **Planning Node:** Before any code is written, the agent analyzes the user's intent, identifies necessary data points, and outlines the logical steps (or joins) required to extract the answer.
2.  **Generation Node:** Translates the approved plan into optimized BigQuery SQL using `Llama-3.1-8B-Instant`.
3.  **Execution Node:** Safely runs the generated SQL against the `fact_trips` table, with built-in error catching to handle invalid syntax.
4.  **Normalization Node:** Converts raw database tuples into structured, human-readable format.
5.  **Reporting Node:** Synthesizes the normalized data into clear business insights, ensuring no hallucinations by grounding the LLM strictly within the provided table context.


## Technical Challenges & Solutions

### 1. The "Hallucination" Trap
* **Challenge:** Database drivers often return unlabelled tuples, causing the LLM to struggle with column association, leading to frequent data misinterpretation.
* **Solution:** Implemented a **Dynamic Header Extraction** engine using regular expressions. By parsing the `AS` aliases directly from the generated SQL, the system enforces a strict contract between the database results and the metadata, ensuring the model always receives a clean, labeled Markdown table.

### 2. The Serialization Mismatch
* **Challenge:** Raw database results were often returned as complex, nested string representations of Python lists or tuples, which caused standard data formatting libraries to fail.
* **Solution:** Developed a robust casting node using `ast.literal_eval` and safe type-casting. This normalized raw database objects into a standardized list-of-lists format, ensuring stable table rendering before ingestion into the LLM context window.



### 3. Production Efficiency (Free Tier Optimization)
* **Challenge:** Operating within the constraints of free-tier token limits and smaller LLM parameter counts (Llama-3.1-8B-Instant).
* **Solution:**
    * **Strategic Token Management:** Implemented data-slicing logic (limiting to the first 10 rows) to keep response payloads well within the 6,000-token limit.
    * **Cost-Effective Architecture:** By tuning prompt engineering for the 8B model instead of scaling to resource-heavy 70B models, I achieved high-quality analytical outputs at a fraction of the compute cost—a design pattern that directly translates to significant savings in high-volume production environments.

## Dataset Information
This project utilizes the **Austin Bikeshare** public dataset, hosted on Google BigQuery (`bi-project-489517.austin_bikeshare.fact_trips`).

* **Domain:** Urban Mobility / Public Transportation.
* **Key Metrics Analyzed:**
    * **Subscriber Usage Patterns:** Trip counts grouped by membership type.
    * **Temporal Analysis:** Average trip duration and weekday morning rush-hour percentage calculations.
    * **Time-Series Trends:** Weekly trip count fluctuations over a rolling 30-day window.

## Key Technical Stack
* **LLM Engine:** Llama-3.1-8B-Instant (via Groq API)
* **Data Warehouse:** Google BigQuery
* **Agentic Framework:** LangGraph / LangChain
* **Deployment:** Streamlit Cloud
* **Data Formatting:** `tabulate` for Markdown-based context injection
