import os
from dotenv import load_dotenv
import streamlit as st
import operator
from typing import TypedDict, Annotated, List, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import os
import json
from google.oauth2 import service_account
from google.cloud import bigquery
from langchain_community.utilities import SQLDatabase
import tempfile

load_dotenv()
api_key = os.getenv("MY_API_KEY")
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
db = get_bigquery_db()

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




# llm = ChatOllama(model="llama3", temperature=0)
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
                                                    ### CRITICAL RULE: FULLY QUALIFIED NAMES
- NEVER write "FROM fact_trips" or "JOIN dim_stations".
- ALWAYS write "FROM `bi-project-489517.austin_bikeshare.fact_trips`".
- Use backticks (`) around the full project.dataset.table path to prevent parsing errors.

BAD: SELECT * FROM dim_stations
GOOD: SELECT * FROM `bi-project-489517.austin_bikeshare.dim_stations`

                                                    

                                                    "When the user doesn't specify a date, calculate metrics for the last 30 days of data available. Use: WHERE trip_date >= DATE_SUB((SELECT MAX(trip_date) FROM fact_trips), INTERVAL 30 DAY)"

                                                    subscriber_id does NOT exist in the dataset.  NEVER include this in your SQL query.
                                                    
                                                    

Division Safety: When calculating percentages, always cast the denominator to prevent integer division: (COUNT(*) * 100.0 / total_trips).
                                                    "STRICT RULE: Never use AT as a table alias. Use full table names or safe aliases like t1, sub_metrics, etc."

                                                

                                                    ### SQL REQUIREMENTS
1. Use explicit aliases for every column in the SELECT statement. 
   Example: AVG(duration_minutes) AS avg_duration_min
2. NEVER use raw calculations or functions without an AS alias.
3. The column aliases MUST NOT contain spaces or special characters (use underscores).
4. Do not include comments inside the ```sql block.

                                                    "In your final SELECT statement, explicitly ALIAS every column with a simple, unique name (e.g., total_trips, rush_pct). Never leave a column as a raw calculation."

                                                    
                                                    Do NOT hallucinate column names, ONLY use the column names present in {schema}.  The column, subscriber_id does NOT EXIST in the schema: {schema}.  Do NOT include subscriber_id in your query.
                                                    
1. CONTRIBUTION METRICS: For "activity" or "share" questions, ALWAYS calculate the percentage based on the Grand Total (SUM), not the Average (AVG).
   - Use a CTE to calculate the total system volume.
   - Use a CROSS JOIN or a second CTE to divide the individual station count by the grand total.
2. NO ESTIMATION: Never guess, simulate, or provide hypothetical numbers.

                                                    CRITICAL
                                                    ONLY SELECT the columns you need to answer the question.  Never select all columns for the final SELECT statement.  Only perform JOINs when absolutely necessary.                                                                                                   
### CORE CALCULATION POLICY (PRIORITY 1)
1. NO ESTIMATION: Never guess, simulate, or provide hypothetical numbers. 
2. SQL-ONLY MATH: All calculations MUST be performed in SQL.  Engineer your own features when necessary to answer the question.
3. FORCED CTE STRUCTURE: For any question involving averages, comparisons, or thresholds:
   a) YOU MUST use WITH clauses (CTEs) to perform the math.
   b) CTE 1: Calculate the base metric per entity (e.g., trips per station).
   c) CTE 2: Calculate the global average or threshold from CTE 1.
   d) FINAL SELECT: Filter or join the entity metrics against the global average.


                                                    BigQuery Scalar Subquery Rule:
When performing arithmetic operations using a value from a Common Table Expression (CTE) or subquery, always ensure the subquery returns a single scalar value of a numeric type (INT64, FLOAT64).

Avoid passing a STRUCT to a math operator. Instead of:
SELECT 100 / (SELECT my_cte FROM my_cte)

Always use explicit column selection to ensure a numeric return:
SELECT 100 / (SELECT column_name FROM my_cte)

Additionally, when calculating percentages, ensure the divisor is cast to FLOAT64 to avoid integer division errors, and use SAFE_DIVIDE(numerator, denominator) to prevent "Division by Zero" crashes.
                                                    

                                                    Aggregation & Safety Rules:

Always Aggregate: Unless specifically asked for "raw data" or a "list," always wrap outlier detection or statistical analysis in a GROUP BY or COUNT(*) to return a summary. Never return thousands of rows of categorical labels (e.g., "normal", "outlier").

Window Functions over Subqueries: Encourage the use of Window Functions (e.g., AVG(col) OVER()) instead of repeated scalar subqueries in the WHERE or CASE clauses to improve BigQuery performance.

Avoid Cartesian Products: Do not comma-join CTEs (e.g., FROM avg_duration, std_duration) as this can create unintended cross-joins. Use a single stats CTE or a CROSS JOIN.

Limit by Default: If a query is likely to return many rows, append LIMIT 100 unless the user specifies otherwise.
                                                    
                                                    When referencing a column that is included in a CTE, you must include the column name in both the CTE AND in your final SELECT statement.  This causes an unrecognized name error in BigQuery.


                                                    SQL Result Integrity Rules:


Enforce Aggregation: If the query includes an aggregate function (like COUNT, AVG, SUM), you must include a GROUP BY clause for all non-aggregated columns.

Prevent Row Multiplication: Always verify that the number of rows in your output matches the expected granularity (e.g., one row per station_id). If you see duplicates, add DISTINCT or ensure the GROUP BY is exhaustive.

Limit Row Count: For exploratory queries, always include LIMIT 50 unless the user explicitly requests a full dataset.

                                                    When referencing specific stations, use station_name instead of station_id.

                                                    Strict Aggregation & Windowing Rules:

No Mixed Aggregation Methods: Never combine GROUP BY clauses with Window Functions (OVER(...)) that contain the same column in the PARTITION BY as the GROUP BY.

Window Function Priority: If you need to calculate per-group statistics (like PERCENTILE_CONT or AVG), use Window Functions exclusively. In this case, perform the calculation in a subquery or CTE without a GROUP BY, then use DISTINCT or a final GROUP BY to collapse the rows.

Aggregation Verification: Every column in your SELECT list must be either:

Wrapped in an aggregate function (e.g., SUM(), AVG(), MIN()).

Included explicitly in the GROUP BY clause.

A constant value.

Standardize Statistics: Prefer calculating metrics via OVER(PARTITION BY ...) in a clean CTE, then selecting from that CTE, rather than forcing a GROUP BY on a query that is already performing windowing.  

                                                    Instead of grouping inside the first CTE, keep the rows unique per station using a DISTINCT or perform the window aggregation first, then select the unique rows.                                             
                                                    You CANNOT query a database for a column it does not have (e.g. total_trips dim_stations).  Create a separate CTE for the column you need to query.

                                                    BigQuery Scalar Value & Type Rules:

Extract, Don't Reference: Never reference a CTE or Subquery directly in an arithmetic expression (e.g., (SELECT val FROM cte) - INTERVAL...). You must alias the column and access it via that alias or ensure the subquery returns a single scalar value.

Correct: (SELECT MAX(date_col) FROM table) - INTERVAL 30 DAY

Correct: (SELECT t.max_date FROM my_cte t) - INTERVAL 30 DAY

Explicit Casting: BigQuery is strictly typed. If a column is a STRING that represents a date, you must explicitly cast it: DATE(my_string_column).

Avoid Implicit Structs: When writing SELECT lists, avoid subqueries that return multiple columns or entire rows. If a subquery only returns one value, always use LIMIT 1 or an explicit column alias to ensure the optimizer treats it as a primitive type rather than a STRUCT.

Use Scalar Subqueries: If you need to use a value from a CTE in a calculation, perform a CROSS JOIN or select the column directly from the CTE to guarantee the engine treats it as a scalar primitive.
                                                    
                                                    Join & Floating Point Rules:

Never Join on Floats: Avoid using FLOAT64 or NUMERIC types in JOIN conditions. These are prone to rounding errors. Always join on INT64, STRING, or DATE keys.

Prefer Window Functions: If you need to filter for the "top" result per category (e.g., longest ride per year), use RANK() OVER (PARTITION BY X ORDER BY Y DESC) in a single CTE instead of performing multiple JOINs on aggregate columns.

Consolidate Aggregates: If a query requires filtering by two different metrics (e.g., top years and top subscribers), calculate all required metrics in one stats CTE, then perform all ranking/filtering in a subsequent step.
                                                    
                                                    Scalar Subquery Enforcement:

No Bare Subqueries in Math: Never use (SELECT col FROM table) as an argument in a function like SAFE_DIVIDE or an arithmetic operator.

Force Scalar Access: Always provide an alias for the subquery and select the column explicitly to guarantee the result is a scalar value.

Bad: SAFE_DIVIDE(col, (SELECT count FROM cte))

Good: SAFE_DIVIDE(col, (SELECT t.count FROM cte t))

Avoid Column Name Conflicts: Do not name a CTE the same as the column within it (e.g., total_trips AS (SELECT COUNT(*) AS total_trips ...)). This causes the SQL engine to return a record type instead of the integer value. Rename the column inside the CTE (e.g., total_val) to avoid this ambiguity.
                                                    
                                                    When you use ORDER BY you NEED to include the column in a GROUP BY clause or wrap it in an aggregate function (e.g., COUNT()).  Without it you will get this error: SELECT list expression references t.total_trips which is neither grouped nor aggregated at [43:7].

When you alias a column in a CTE, it only lives inside that CTE.  You cannot access the alias in another SELECT statement without explicitly JOINing to the original CTE. However, you should follow best practices which are to calculate the metric and alias it (with the same name) in the same CTE you are SELECTing from. 
Strict UNION ALL Compliance:

Before executing a UNION ALL, verify that both SELECT statements have the exact same number of columns, in the exact same order, and with identical data types.

If you are joining disparate datasets (e.g., Stations vs. Subscribers), cast all columns to a common type (usually STRING or FLOAT64) using CAST(column AS STRING).

Use NULL AS [missing_column_name] as a placeholder if one dataset lacks a field present in the other, ensuring column alignment.
If top_stations has 6 columns but top_subscriber_types has 5, or if you accidentally swapped the order (e.g., subscriber_type is the 3rd column in one and the 4th in the other), BigQuery will fail with an error—often pointing to the mismatch in the second query.

ALL columns SELECTed in UNION ALL need to exist in the CTE you are querying or JOINed.  If this is not done, the query will fail because the column you are SELECTing for does not exist in that CTE.
                                                    When you create a subquery, the table only lives within that subquery.  If you have create a CTE and try to SELECT from a table created with a subquery, you will NOT be able to access anything but the aliased name from the subquery.  In a CTE, do NOT attempt to perform a calculation using a column SELECTed for from within a subquery.  In these cases, ALWAYS use the aliased name from the subquery.

NEVER create duplicate CTEs.  Define a CTE ONLY ONCE when writing queries.                                                   
                                                    EXTRACT does not support literal STRING arguments.  Do not pass literal strings to the EXTRACT function.
                                                    
                                                    Avoid UNION ALL if possible.  DON't use UNION ALL unless there are no other alternatives.

                                                    Use station_name instead of station_id when responding to a user query asking about specific stations.                                                

                                                    AT is a RESERVED KEYWORD in BigQuery, NEVER use 'at' at all when generating your queries.  It results in this error: Syntax error: Expected end of input but got keyword AT at [55:17]


                                                    CRITICAL SQL MANDATE: Redundant Extraction Protocol
"When constructing SQL queries involving CTEs, you must adhere to the Redundant Extraction Rule:

Isolation Principle: CTEs are entirely isolated sandboxes. They cannot 'see' aliases created in other CTEs.

No Cross-CTE Aliasing: You are strictly forbidden from referencing an alias defined in one CTE within another CTE.

Mandatory Re-Calculation: Every CTE must be self-contained. If you need a column like trip_year (which requires EXTRACT(YEAR FROM trip_date)), you must perform the full EXTRACT calculation inside the SELECT clause of every single CTE that uses that data.

DON't do this: avg_trip_time AS (SELECT trip_year, ...) (This will fail).

Example of REQUIRED behavior: avg_trip_time AS (SELECT EXTRACT(YEAR FROM trip_date) AS trip_year, ...) (This will succeed).

Verification: Before finishing a query, verify that every column used in a SELECT, GROUP BY, or JOIN clause is either a raw column from the source table or a new calculation/alias created within that specific block.
---
Generate only the final SQL query in a ```sql ... ``` block.
Generated SQL:
""")

# Response generator remains largely the same but simplified
RESPONSE_GENERATOR_PROMPT = PromptTemplate.from_template("""
User question: {question}
Result: {db_result}

Provide a clear, professional answer. If the result is empty, inform the user that no data matches their criteria.  If the user asks an analytical question, do NOT provide a textual explanation or hypothetical calculation.  Only use the information you find from querying the database when formulating your response.  When listing, use a bullet point format.  Return the provided data and report it, (e.g., "Here's the data you asked for"). Contexualize or explain the results only if required to answer the user's question. Do not provide code blocks.
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
    hardcoded_schema = "Tables: fact_trips (trip_id, start_station_id, end_station_id, duration_minutes, trip_date, start_hour, subscriber_type, bike_id, end_station_id), dim_stations (station_id, station_name, status, location), dim_subscribers (subscriber_category, subscriber_type)"
    response = sql_generator_chain.invoke({
        "plan": state["intermediate_steps"][-1].content.replace("PLAN: ", ""),
        "question": state["question"],
        "schema": hardcoded_schema,
        "error_context": error or "None"
    }).content

    # Strict regex to extract only the SQL code block
    match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
    sql_query = match.group(1).strip() if match else response.strip()
    try:
        return {"sql_query": sql_query, "error": None}
    except Exception as e:
        print(f"--- SQL ERROR: {str(e)} ---")
        return {"sql_query": None, "error": str(e)}



import ast
from tabulate import tabulate

def validate_and_execute_sql(state: AgentState):
    sql_query = state.get("sql_query", "")
    current_attempt = state.get("attempt", 0)
    
    if not db:
        return {"error": "Database connection is missing.", "attempt": current_attempt + 1}

    try:
        # 1. Execute via SQLAlchemy engine to get real metadata
        from sqlalchemy import text
        
        with db._engine.connect() as connection:
            result = connection.execute(text(sql_query))
            # Extract the actual column names from the cursor
            headers = list(result.keys())
            # Fetch all rows
            raw_result = result.fetchall()

        # 2. Handle empty results
        if not raw_result:
            return {"db_result": "No data found", "error": None, "attempt": current_attempt}

        # 3. Create Markdown table using the real headers from BigQuery
        formatted_table = tabulate(
            raw_result, 
            headers=headers, 
            tablefmt="pipe",
            showindex=False
        )
            
        return {"db_result": formatted_table, "error": None, "attempt": current_attempt}        
        
    except Exception as e:
        print(f"--- DB ERROR (Attempt {current_attempt + 1}): {str(e)} ---")
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
    You are a Senior Data Scientist. You are provided with a Markdown table representing results from the Austin Bikeshare database.

    ---
    DATA TABLE:
    {db_result}
    ---

    INSTRUCTIONS:
    1. Answer the user's question: "{state['question']}"
    2. Use a professional, conversational tone.
    3. If percentages or averages are provided in the table, report them exactly as shown. Do not recalculate them unless the math is explicitly requested.
    4. If a user asks for which year was the highest average trip count, the count for the year with the highest number of rides may be repeated in the results, IGNORE this.  Treat the repeated value as one single value, which you report to the user.
    5. When reporting, round to the second decimal place.
    5. Take your time to analyze the data and provide a clear and concise response to the user.

    Answer:"""
    
    # Execution using the LLM 
    try:
        final_answer = llm.invoke(prompt).content
        return {"final_answer": final_answer}
    except Exception as e:
        return {"final_answer": f"Data retrieved successfully, but I had trouble formatting the summary. Raw Data: {db_result} Error: {e}"}
    

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
