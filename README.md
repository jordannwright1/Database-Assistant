# DataAgent: Autonomous SQL Execution for BigQuery
[Click Here To Test The Agent](https://database-assistant-3brgrkrmo3vkpwtsqfr87r.streamlit.app)

DataAgent is an autonomous SQL generation and execution system that bridges the gap between natural language and structured BigQuery data. Designed for extreme efficiency, the system leverages a compact, high-performance LLM architecture to translate user queries into valid SQL, execute them, and interpret the results with high analytical accuracy.



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
