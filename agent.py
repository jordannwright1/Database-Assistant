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
api_key = os.getenv("MY_API_KEY")
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
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("MY_API_KEY") 
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
You are a precise Data Analyst. You write SQL for BigQuery.
User question: {question}
Plan: {plan}
Database Schema: {schema}
Error Context: {error_context}

### CORE CALCULATION POLICY (PRIORITY 1)
1. NO ESTIMATION: Never guess, simulate, or provide hypothetical numbers. 
2. SQL-ONLY MATH: All calculations ("average", "ranking", "above/below average") MUST be performed in SQL.  Engineer your own features when necessary to answer the question, such as for calculating average trips per station.
3. FORCED CTE STRUCTURE: For any question involving averages, comparisons, or thresholds:
   a) YOU MUST use WITH clauses (CTEs) to perform the math.
   b) CTE 1: Calculate the base metric per entity (e.g., trips per station).
   c) CTE 2: Calculate the global average or threshold from CTE 1.
   d) FINAL SELECT: Filter or join the entity metrics against the global average.

### SQL RULES (PRIORITY 2)
1. RESERVED KEYWORDS: If a column or alias is a reserved keyword (like 'AT'), wrap it in backticks: `AT`.
2. GROUPING: Always group by ID. Use ANY_VALUE(name_column) for descriptive names in SELECT.
3. SOURCE: Prefer vw_trips_detailed unless specific joins are required.
4. NO DML: Never use INSERT, DELETE, DROP, or ALTER.

### SCHEMA LOCK
- fact_trips: trip_id, start_station_id, end_station_id, subscriber_id, duration_minutes, start_hour, trip_date.
- dim_stations: station_id, station_name.
- dim_subscribers: subscriber_id, subscriber_category.
- ALWAYS JOIN dim_stations ON start_station_id = station_id.
- ALWAYS JOIN dim_subscribers ON subscriber_id = subscriber_id.

---
Generate only the final SQL query in a ```sql ... ``` block.
Generated SQL:
""")

# Response generator remains largely the same but simplified
RESPONSE_GENERATOR_PROMPT = PromptTemplate.from_template("""
User question: {question}
Result: {db_result}

Provide a clear, professional answer. If the result is empty, inform the user that no data matches their criteria.  If the user asks an analytical question, do NOT provide a textual explanation or hypothetical calculation.  Only use the information you find from querying the database when formulating your response.  When listing, use a bullet point format.  Return the provided data and report it, (e.g., "Here's the data you asked for"). Contexualize or explain the results only if required to answer the user's question. 
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
    error = state.get("error")
    # Only allow the LLM to output the SQL code block
    sql_generator_chain = SQL_GENERATOR_PROMPT | llm
    
    response = sql_generator_chain.invoke({
        "plan": state["intermediate_steps"][-1].content.replace("PLAN: ", ""),
        "question": state["question"],
        "schema": db.get_table_info(),
        "error_context": error or "None"
    }).content

    # Strict regex to extract only the SQL code block
    match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
    sql_query = match.group(1).strip() if match else response.strip()
    
    return {"sql_query": sql_query, "error": None}


def validate_and_execute_sql(state: AgentState):
    sql_query = state.get("sql_query", "")
    current_attempt = state.get("attempt", 0)
    
    # 1. Guardrail
    forbidden = ["DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE"]
    if any(keyword in sql_query.upper() for keyword in forbidden):
        return {"error": "Destructive SQL detected.", "db_result": "BLOCKED", "attempt": current_attempt + 1}
    
    try:
        # 2. Execution
        print(f"--- Executing SQL (Attempt {current_attempt + 1}): {sql_query} ---")
        db_result = db.run(sql_query) 
        
        # 3. Truncate
        result_str = str(db_result)
        if len(result_str) > 2000:
            result_str = result_str[:2000] + "..."
            
        return {"db_result": result_str, "error": None} # Clear error on success
        
    except Exception as e:
        print(f"--- DB ERROR: {str(e)} ---")
        # Return error and INCREMENT the attempt
        return {"error": str(e), "db_result": "ERROR", "attempt": current_attempt + 1}


def respond_to_user(state: AgentState):
    # Ensure the result is formatted as a clear string
    db_result = state.get("db_result")
    
    prompt = f"""
    You are a Data Assistant. Use the provided DATABASE RESULT to answer the user's question.
    
    1. Look specifically for the column or value representing 'average_trips_per_station' if included in the user query.
    2. If the data contains an average value, state it clearly in your answer.
    3. Read the {db_result} carefully and answer the user's question.
    
    Question: {state['question']}
    DATABASE RESULT: {db_result}
    
    Answer:"""
    
    final_answer = llm.invoke(prompt).content
    return {"final_answer": final_answer}

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
