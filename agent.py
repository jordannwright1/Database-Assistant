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
from google.oauth2 import service_account
from google.cloud import bigquery

load_dotenv()
api_key = os.getenv("MY_API_KEY")

# 1. Force the Google client to NOT look for the metadata server
# This prevents the initial timeout before we even define the client
os.environ["GOOGLE_AUTH_DISABLE_METADATA"] = "1"

# 2. Extract and format the credentials
# Note: Use the exact key names as they appear in your TOML
sa_info = dict(st.secrets["gcp_service_account"])
credentials = service_account.Credentials.from_service_account_info(sa_info)

# 3. Explicitly construct the BigQuery client
# Adding the client_options explicitly locks the communication channel
client = bigquery.Client(
    credentials=credentials, 
    project=sa_info["project_id"],
    client_options={"api_endpoint": "https://bigquery.googleapis.com"}
)
# 3. Create the BigQuery Client explicitly
# This is the "kill switch" for the TransportError
custom_bq_client = bigquery.Client(
    credentials=credentials, 
    project=sa_info.get("project_id", "bi-project-489517")
)

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
    attempts = state.get("attempt", 0)
    
    # 1. If we have a hard error
    if state.get("error"):
        # If we have reached 2 attempts, do not return "generate_sql"
        if attempts >= 2:
            return "respond" 
        return "generate_sql"
    
    # 2. Success path
    return "respond"


# --- 2. Database Connection & Semantic Layer ---
# Initialize BigQuery connection (read-only service account recommended)
db = SQLDatabase.from_uri(
    "bigquery://bi-project-489517/austin_bikeshare",
    engine_args={"connect_args": {"client": custom_bq_client}}
)# llm = ChatOllama(model="llama3", temperature=0)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", temperature=0)

llm_smart = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("MY_API_KEY") 
)

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
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
You are a precise Senior BigQuery Architect. You write cost-efficent, mathematically precise SQL for BigQuery.
User question: {question}
Plan: {plan}
Database Schema: {schema}
Error Context: Use {error_context} to modify the original SQL query to resolve the error.

PROJECT: bi-project-489517
DATASET: austin_bikeshare
ALL FROM statements MUST reference the PROJECT and DATASET explicitly, NEVER query just the table name (e.g., fact_trips).  

                                                    If more information is required, perform JOINS to connect to the dim tables in FIRST SELECT statement and find the information you need to engineer features that calculate the values the user requests in the query.  The first CTE should include all of the necessary JOIN statements to the dim tables (e.g. dim_stations, etc.) needed in order to answer the question {question}.

                                                    "Never use CURRENT_DATE or hardcoded dates. Always use (SELECT MAX(trip_date) FROM fact_trips) as the anchor point for your WHERE clauses."

                                                    "STRICT RULE: Never use AT as a table alias. Use full table names or safe aliases like t1, sub_metrics, etc."

                                                    "Constraint: When creating CTEs, ensure every column required for subsequent aggregations is explicitly selected in the prior CTE's SELECT statement."

                                                    ### SQL REQUIREMENTS
1. Use explicit aliases for every column in the SELECT statement. 
   Example: AVG(duration_minutes) AS avg_duration_min
2. NEVER use raw calculations or functions without an AS alias.
3. The column aliases MUST NOT contain spaces or special characters (use underscores).
4. Do not include comments inside the ```sql block.

                                                    "In your final SELECT statement, explicitly ALIAS every column with a simple, unique name (e.g., total_trips, rush_pct). Never leave a column as a raw calculation."                                                
### CORE CALCULATION POLICY (PRIORITY 1)
1. NO ESTIMATION: Never guess, simulate, or provide hypothetical numbers. 
2. SQL-ONLY MATH: All calculations MUST be performed in SQL.  Engineer your own features when necessary to answer the question.
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



import ast
from tabulate import tabulate

def validate_and_execute_sql(state: AgentState):
    sql_query = state.get("sql_query", "")
    current_attempt = state.get("attempt", 0)
    
    try:
        # 1. Execution
        raw_result = db.run(sql_query)
        
        # Ensure we have a list of tuples/lists
        if isinstance(raw_result, str):
            try:
                raw_result = ast.literal_eval(raw_result)
            except:
                pass
        
        if not raw_result or raw_result == "[]":
            return {"db_result": "No data found", "error": None, "attempt": current_attempt}

        headers = re.findall(r"AS\s+([a-zA-Z0-9_]+)", sql_query, re.IGNORECASE)
        
        # If extraction fails, fallback to readable placeholders
        if not headers or len(headers) != len(raw_result[0]):
            headers = ["membership_type", "avg_duration_min", "total_trips", "rush_trips", "rush_percentage"]
        # 3. Create Markdown table
        formatted_table = tabulate(
            raw_result, 
            headers=headers, 
            tablefmt="pipe",
            showindex=False
        )
            
        return {"db_result": formatted_table, "error": None, "attempt": current_attempt}        
        
    except Exception as e:
        print(f"--- DB ERROR (Attempt {current_attempt + 1}): {str(e)} ---")
        # Increment the attempt counter to trigger the 2-try limit
        return {
            "error": str(e), 
            "db_result": "ERROR", 
            "attempt": current_attempt + 1 
        }
    

def respond_to_user(state: AgentState):
    db_result = state.get("db_result")
    error = state.get("error")
    attempt = state.get("attempt", 0)
    print(f"DEBUG:{db_result}")
    # 1. Handle Critical Failures (The "2-Attempt" Exit)
    if error and attempt >= 2:
        return {"final_answer": f"I attempted to resolve the data request twice but encountered a persistent error: {error}. Please try rephrasing your question."}

    # 2. Handle No Data
    if not db_result or db_result == "No data found":
        return {"final_answer": "The query was successful, but no data was found matching your specific criteria in the Austin Bikeshare dataset."}

    # 3. Enhanced Prompting for Data Accuracy
    prompt = f"""
    You are a Senior Data Analyst. You are provided with a Markdown table representing results from the Austin Bikeshare database.

    ---
    DATA TABLE:
    {db_result}
    ---

    INSTRUCTIONS:
    1. Answer the user's question: "{state['question']}"
    2. Use a professional, conversational tone.
    3. If percentages or averages are provided in the table, report them exactly as shown. Do not recalculate them unless the math is explicitly requested.

    Answer:"""
    
    # Execution using the LLM 
    try:
        final_answer = llm.invoke(prompt).content
        return {"final_answer": final_answer}
    except Exception as e:
        return {"final_answer": f"Data retrieved successfully, but I had trouble formatting the summary. Raw Data: {db_result}"}
    

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
