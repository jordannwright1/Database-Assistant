import operator
from typing import TypedDict, Annotated, List, Union
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import tempfile
import json

@st.cache_resource
def get_bigquery_db():
    # 1. Handle Cloud Deployment
    if "gcp_service_account" in st.secrets:
        # Create a temporary file to hold the secrets safely
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(dict(st.secrets["gcp_service_account"]), f)
            temp_path = f.name
        
        # Connect using the temporary file path
        db_uri = f"bigquery://bi-project-489517/austin_bikeshare?credentials_path={temp_path}"
    
    # 2. Handle Local Development
    else:
        db_uri = "bigquery://bi-project-489517/austin_bikeshare"
        
    return SQLDatabase.from_uri(db_uri)

load_dotenv()
creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
api_key = os.getenv("GROQ_API_KEY")

# --- 1. Define Agent State ---
class AgentState(TypedDict):
    question: str
    intermediate_steps: Annotated[List[str], operator.add] # Changed from BaseMessage to str
    sql_query: str
    db_result: str
    final_answer: str
    error: str
    attempt: int

def should_continue(state: AgentState):
    # If there is an error, check if we should retry
    if state.get("error"):
        current_attempt = state.get("attempt", 0)
        if current_attempt < 2: # Allow 2 attempts total
            return "generate_sql"
        else:
            return "respond" # Go to respond node to report the error to user
    
    # Otherwise, proceed to respond
    return "respond"

# --- 2. Database Connection & Semantic Layer ---
# Initialize BigQuery connection (read-only service account recommended)
db = get_bigquery_db()
# llm = ChatOllama(model="llama3", temperature=0)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", temperature=0)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=api_key
)
llm_smart = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key 
)

# --- 3. Prompts ---
PLANNER_PROMPT = PromptTemplate.from_template("""
You are an expert Data Analyst. Your goal is to answer the user's question using the Austin Bikeshare dataset.

---
### DATABASE SCHEMA DEFINITION:
You are querying a Star Schema database with the following structure:

fact_trips: The central fact table containing trip_id, start_station_id, end_station_id, subscriber_id, bike_id, duration_minutes, start_hour, and trip_date.

dim_stations: Contains station_id, station_name, latitude, and longitude.

dim_subscribers: Contains subscriber_id, subscriber_category, and other user demographic metadata.

REQUIRED JOIN LOGIC:

Always join fact_trips to dim_stations on start_station_id = station_id.

Always join fact_trips to dim_subscribers on subscriber_id = subscriber_id.

Use these joins for all requests involving station names or subscriber categories.

---
### Your Decision Process:
1. **Scope:** Identify if a date filter is needed. Default to the most recent data if unspecified.
2. **Strategy:** - If the question can be answered by `vw_trips_detailed`, create a plan that selects from that view.
   - If the question requires specific joins (e.g., matching stations with specific statuses), explicitly name the tables and the JOIN keys (e.g., 'fact_trips.start_station_id = dim_stations.station_id').
3. **Metric Rules:**
   - Average duration: 'AVG(duration_minutes)'.
   - Trip Count: 'COUNT(trip_id)'.
   - Conversion/Subscriber Rate: 'COUNT(trip_id) / SUM(COUNT(trip_id)) OVER()'.
4. **Data Cleaning:** Always use `WHERE` clauses to filter out invalid or null keys (e.g., `WHERE start_station_id IS NOT NULL`).
                                              
### SCHEMA INTERDEPENDENCY RULE:
Before formulating a query, check if the request requires user or station attributes.

If the user asks for subscriber_category or subscriber_type, you must plan to join fact_trips to dim_subscribers on subscriber_id.

If the user asks for station_name, you must plan to join fact_trips to dim_stations on start_station_id.

Never assume these columns exist in fact_trips. Your plan must explicitly state: 'Perform JOIN with [dimension table] on [key].                                             

Your response should be 'PLAN: [Your detailed plan]'.
""")

SQL_GENERATOR_PROMPT = PromptTemplate.from_template("""
### ROLE
You are a Senior BigQuery Architect. Write high-performance, one-shot SQL queries.

### THE "STATION SUMMARY" PATTERN (MANDATORY)
To compare multiple stations accurately without duplicate rows:
1. SELECT DISTINCT: Always use `SELECT DISTINCT` in the outermost layer.
2. NESTED AGGREGATION: 
   - Inner subquery: `COUNT(*)` grouped by `station_id` and `trip_date`.
   - Outer layer: Calculate `PERCENTILE_CONT` partitioned by `station_id`.
3. SCOPE: Refer to columns only by their names in the outer layer (e.g., `station_id`, not `s.station_id`).

### MANDATORY SQL TEMPLATE
SELECT DISTINCT
  station_name,
  station_id,
  PERCENTILE_CONT(daily_count, 0.5) OVER(PARTITION BY station_id) AS median_daily_trips,
  MAX(daily_count) OVER(PARTITION BY station_id) / NULLIF(AVG(daily_count) OVER(PARTITION BY station_id), 0) AS outlier_ratio
FROM (
  SELECT 
    s.station_name, 
    base.station_id, 
    base.daily_count
  FROM (
    SELECT start_station_id AS station_id, trip_date, COUNT(*) as daily_count
    FROM `bi-project-489517.austin_bikeshare.fact_trips`
    WHERE trip_date >= DATE_SUB((SELECT MAX(trip_date) FROM `bi-project-489517.austin_bikeshare.fact_trips`), INTERVAL 30 DAY)
    GROUP BY 1, 2
  ) base
  JOIN `bi-project-489517.austin_bikeshare.dim_stations` s ON base.station_id = s.station_id
)
ORDER BY median_daily_trips DESC
LIMIT 10;

### SCHEMA INFO
{schema}

Question: {question}
Plan: {plan}
Previous Error (if any): {error_context}

Generate only the SQL in a ```sql ... ``` block.
""")


# --- 4. Nodes (Functions for each step in the graph) ---
def plan_and_disambiguate(state: AgentState):
    planner_chain = PLANNER_PROMPT | llm
    plan_output = planner_chain.invoke({"question": state["question"]})
    
    # Return just the string content to keep the state "hashable"
    return {
        "intermediate_steps": [str(plan_output.content)]
    }


def generate_sql(state: AgentState):
    error = state.get("error", "None")
    current_attempt = state.get("attempt", 0) + 1
    
    # Access the last string in intermediate_steps
    last_plan = state["intermediate_steps"][-1] if state["intermediate_steps"] else ""
    
    formatted_prompt = SQL_GENERATOR_PROMPT.format(
        question=state['question'],
        plan=last_plan.replace("PLAN: ", ""),
        schema=db.get_table_info(),
        error_context=error
    )
    
    response = llm.invoke(formatted_prompt).content
    
    match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
    sql_query = match.group(1).strip() if match else response.strip()
    return {"sql_query": sql_query, "error": None, "attempt": current_attempt}



import ast
import re
from tabulate import tabulate

def validate_and_execute_sql(state: AgentState):
    sql_query = state.get("sql_query", "")
    print(f"--- EXECUTING SQL ---")
    
    try:
        raw_result = db.run(sql_query, fetch="all")
        
        # 1. Handle stringified results
        if isinstance(raw_result, str):
            try:
                raw_result = ast.literal_eval(raw_result)
            except:
                return {"db_result": str(raw_result), "error": None}
        
        if not raw_result:
            return {"db_result": "No data found", "error": None}

        # 2. Robust dict-to-list conversion
        processed_data = []
        for row in raw_result:
            if isinstance(row, dict):
                processed_data.append(list(row.values()))
            else:
                processed_data.append(row)

        # 3. Headers extraction
        header_match = re.findall(r"AS\s+[`'\" ]*([a-zA-Z0-9_]+)|SELECT\s+([a-zA-Z0-9_.]+)", sql_query, re.IGNORECASE)
        headers = [h[0] if h[0] else h[1].split('.')[-1] for h in header_match]
        
        # 4. Generate table
        formatted_table = tabulate(
            processed_data[:10], 
            headers=headers[:len(processed_data[0]) if processed_data else 0], 
            tablefmt="pipe"
        )
        
        # CRITICAL FIX: Ensure return is a simple dictionary
        # Explicitly ensuring no complex objects leak into the state
        return {
            "db_result": str(formatted_table), 
            "error": None
        }

    except Exception as e:
        # Catch and return error as a string
        return {"db_result": "ERROR", "error": str(e)}



def respond_to_user(state: AgentState):
    db_result = state.get("db_result", "No results found.")
    error_context = state.get("error", "")
    
    # 1. CRITICAL ERROR HANDLING: 
    # If the execution node flagged an error, stop here and report it.
    if db_result == "ERROR" or (error_context and "failed" in error_context.lower()):
        error_message = f"I encountered a technical error while retrieving the data: {error_context}"
        return {"final_answer": error_message}

    # 2. NO DATA HANDLING:
    if db_result == "No data found" or not db_result:
        return {"final_answer": "The query executed successfully, but no records matched your request in the Austin Bikeshare dataset."}

    # 3. PROMPT PREPARATION (Keep your exact prompt structure)
    prompt = f"""
        You are a Data Reporter and Journalist. You are provided with a final, filtered SQL result table which you must report to the user with 100% accuracy.
        
        DATABASE RESULT:
        {db_result}
        
        CORE GUIDELINES:
        1. TRUST THE DATA: The data has already been aggregated. Do not re-filter or question the source.
        2. RANKING INTEGRITY: You MUST report stations in the order they appear in the table. If 'Nueces & 26th' has a higher number than '6th & Congress', it must be listed first.
        3. AVOID DATA HALLUCINATION: Use exact values from the table. If a metric isn't there, state: 'The provided data does not contain the specific metrics to calculate this value.'
        4. COLUMN MAPPING: Use the provided columns to answer the question. You do not need to include technical column names in your prose.
        5. PRESENTATION: Use a professional, conversational tone. Use bullet points for lists and tables where they improve clarity. Do not use nested bullet points or generate code blocks.

        6. SORTING VERIFICATION: When reporting top stations, verify the values manually. 
        If the table values are [3838, 3798, 3619], you MUST list them in that exact order (3838 first, then 3798, then 3619).

        STATISTICAL INTERPRETATION (CRITICAL):
        - OUTLIER RATIO: This is defined as (Max Daily Trips / Average Daily Trips). 
        - A ratio of 1.0 means perfectly consistent daily volume.
        - A high ratio (e.g., > 10.0) indicates a massive one-day "spike" or anomaly relative to that station's typical activity.
        - NEVER mention 'Interquartile Range' (IQR) or other statistical theories not present in the data.

        RESPONSE STRUCTURE:
        1. Directly answer the user's question using the Top results from the table.
        2. Highlight any significant outliers based on the 'outlier_ratio'.
        3. Provide a brief conversational summary of what these trends suggest about the Austin Bikeshare network.

        Question: {state['question']}
        
        Answer:"""
    
    # 4. EXECUTION
    print(f"DEBUG DATA: {db_result}")
    
    try:
        # Use your standard llm invocation
        response = llm.invoke(prompt).content
        return {"final_answer": response}
    except Exception as e:
        print(f"--- RESPONSE NODE ERROR: {str(e)} ---")
        return {"final_answer": f"I analyzed the data, but I'm having trouble formatting the response right now. Raw Result: {db_result}"}
    

def decide_next_step(state: AgentState):
    if state["clarification_needed"]:
        return "end_clarification" # Or a dedicated node for user interaction
    if state["error"] and state["attempt"] < 1: # Retry once
        return "generate_sql" # Go back to generate SQL
    if state["final_answer"]:
        return "end_response"
    return "execute_sql" # Default to executing if no explicit end or error

# --- 5. Build the LangGraph ---
workflow = StateGraph(AgentState)

# 1. ADD NODES
workflow.add_node("plan", plan_and_disambiguate)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", validate_and_execute_sql) 
workflow.add_node("respond", respond_to_user)

# 2. DEFINE EDGES
workflow.set_entry_point("plan")
workflow.add_edge("plan", "generate_sql")
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_conditional_edges("execute_sql", should_continue, {
    "generate_sql": "generate_sql", 
    "respond": "respond"
})
workflow.add_edge("respond", END)

app = workflow.compile()
