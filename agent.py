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
# --- 1. Define Agent State (The "Memory" of the Graph) ---
class AgentState(TypedDict):
    question: str
    intermediate_steps: Annotated[List[BaseMessage], operator.add]
    sql_query: str
    db_result: str
    final_answer: str
    error: str
    attempt: int 

def should_continue(state: AgentState):
    # 1. If we have a hard error, try to fix it
    if state.get("error"):
        if state.get("attempt", 0) >= 2:
            return "respond" # Give up and tell user why
        return "generate_sql"
    
    # 2. If the result is empty/null, it's a data failure
    if not state.get("db_result") or state.get("db_result").strip() in ["", "[]", "None"]:
        # Optional: Increment attempt and retry or report to user
        return "respond" 
        
    # 3. Success path
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
You are a Senior BigQuery Architect. Write high-performance, single-pass SQL.

### MANDATORY DATA STRUCTURE
Every query MUST return a clear table with these rules:
1. IDENTIFIER: The first column must always be the grouping variable (e.g., `subscriber_type` or `day_of_week`).
2. DIRECT AGGREGATION: Use `AVG(CASE WHEN...)` or `COUNTIF(...)` directly in the SELECT.
3. NO CTEs: Write the query as one block: `SELECT... FROM... WHERE... GROUP BY...`.
4. NO TRUNCATION: Complete the entire query. Do not stop mid-sentence.

### CALCULATION FORMULAS
- Duration: `AVG(duration_minutes) AS avg_duration`
- Percentage: `(COUNTIF(condition) * 100.0 / COUNT(*)) AS metric_pct`
- Comparison: `(AVG(CASE WHEN A) - AVG(CASE WHEN B)) AS discrepancy`

### SQL CONSTRAINTS
- Table: `bi-project-489517.austin_bikeshare.fact_trips`
- Date: `trip_date >= DATE_SUB((SELECT MAX(trip_date) FROM ...), INTERVAL 30 DAY)`

Question: {question}
Generate only the SQL in a ```sql ... ``` block.
""")


# --- 4. Nodes (Functions for each step in the graph) ---
def plan_and_disambiguate(state: AgentState):
    planner_chain = PLANNER_PROMPT | llm
    plan_output = planner_chain.invoke({"question": state["question"]})
    
    if "CLARIFICATION_NEEDED" in plan_output.content:
        return {"clarification_needed": True, "final_answer": plan_output.content}
    else:
        return {"clarification_needed": False, "intermediate_steps": [AIMessage(content=plan_output.content)]}

import re

def generate_sql(state: AgentState):
    error = state.get("error", "None")
    
    # Fill the template
    formatted_prompt = SQL_GENERATOR_PROMPT.format(
        question=state['question'],
        plan=state["intermediate_steps"][-1].content.replace("PLAN: ", ""),
        schema=db.get_table_info(),
        error_context=error
    )
    
    # Invoke
    response = llm_smart.invoke(formatted_prompt).content
    
    # Extract SQL
    match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
    sql_query = match.group(1).strip() if match else response.strip()
    
    return {"sql_query": sql_query, "error": None}


from tabulate import tabulate


def validate_and_execute_sql(state: AgentState):
    sql_query = state.get("sql_query", "")
    try:
        raw_result = db.run(sql_query)
        import ast
        results = ast.literal_eval(raw_result) if isinstance(raw_result, str) else raw_result
        
        if not results:
            return {"db_result": "No data found", "error": None}

        # Find the VERY LAST Select block to get the final aliases
        final_select_part = sql_query.split('FROM')[-2].split('SELECT')[-1]
        headers = [h.split(' AS ')[-1].strip().replace('`', '').split('.')[-1] 
                   for h in final_select_part.split(',')]

        formatted_result = tabulate(results, headers=headers, tablefmt="pipe")
        
        return {"db_result": formatted_result, "error": None}
    except Exception as e:
        return {"db_result": "ERROR", "error": str(e)}
    
def respond_to_user(state: AgentState):
    db_result = state.get("db_result", "No results found.")
    
    # Automatically extract headers if result is in list/table format
    header_hint = "Columns found in result: " + (db_result.split('\n')[0] if '\n' in db_result else "Unknown")
    if not db_result or db_result == "ERROR":
        return {"response": "I encountered an error retrieving data. Please check the query logs."}
    prompt = f"""
    You are a data reporter. You are provided with the final, filtered result of a SQL query, which you will report to the user.
    
    {header_hint}
    
    DATABASE RESULT:
    {db_result}
    
    CORE GUIDELINES:
    1. TRUST THE DATA: The data provided has already been filtered, joined, and aggregated by the SQL query. Do not attempt to re-filter, calculate missing time periods, or question the source of the data.
    2. AVOID DATA HALLUCINATION: If the columns (like 'total_trip_count' or 'average_trip_duration') are present in the table above, the data is complete. Do not claim information is missing.
    3. COLUMN MAPPING: Map the columns provided in the header hint to the user's question. Use these exact values.  You don't need to include the column names in your response.
    4. PRESENTATION: Use the data provided to answer the user's question in a conversational and professional manner. You may add context only when required to answer the user's question.
    5. RESPONSE: Read the {db_result} carefully and respond to the user's question.
    6. CONTEXTUALIZE: Only use the information you find from querying the database when formulating your response. When listing, use a bullet point format and clearly separate each bullet, do not include bullet points within bullet points.    You may also use tables when appropriate.  Do not generate addtional code blocks in your response. 
    7. CRITICAL: You are NOT allowed to invent data. If a specific metric is not in the {db_result} provided to you, you must state: 'The provided data does not contain the specific metrics to calculate this value.  NEVER make assumptions about the data. 
    
    8. AFTER you present the data, you may provide a conversational response only if it helps to answer the question.

    Question: {state['question']}
    
    Answer:"""
    print(f"DEBUG DATA: {state['db_result']}")
    response = llm.invoke(prompt).content
    return {"final_answer": response}


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
