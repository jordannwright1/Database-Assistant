import streamlit as st
import uuid
from agent import app 
import os 
from groq import Groq
# Bridge secrets to environment variables
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
client = Groq()

# --- 1. Page Configuration ---
st.set_page_config(page_title="Austin Bikeshare Dataset AI Assistant", page_icon="📊", layout="wide")

st.title("📊 Austin Bikeshare Dataset AI Assistant")
st.markdown("""
This agent uses **LangGraph** to brainstorm plans, generate BigQuery SQL, 
and extract insights from the Austin Bikeshare dataset.
""")

# --- 2. Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Unique thread ID for the LangGraph state memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- 3. Sidebar: Database Schema Reference ---
with st.sidebar:
    st.header("Schema Info")
    st.info("""
    **Simplified Access:**
    - `vw_trips_detailed` (Recommended for most queries)

    **Fact Table:**
    - `fact_trips`: `trip_id`, `start_station_id`, `end_station_id`, `duration_minutes`, `trip_date`

    **Dimension Tables:**
    - `dim_stations`: `station_id`, `station_name`, `status`
    - `dim_subscribers`: `subscriber_type`, `subscriber_category`
    """)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 4. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Only show SQL if it exists in the message object
        if message.get("sql"):
            st.code(message["sql"], language="sql")

# --- 5. User Input & Agent Logic ---
if prompt := st.chat_input("Ex: How does the average trip duration for 'Annual Membership' subscribers compare to 'Pay-as-you-ride' users during morning rush hour (7 AM to 9 AM) versus mid-day (11 AM to 3 PM)?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        sql_placeholder = st.empty()
        response_placeholder = st.empty()
        
        inputs = {"question": prompt, "attempt": 0}
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        final_response = ""
        generated_sql = ""

        with st.spinner("Analyzing..."):
            for output in app.stream(inputs, config=config):
                for node_name, state_update in output.items():
                    if node_name == "plan":
                        status_placeholder.markdown("🧠 *Brainstorming query plan...*")
                    elif node_name == "generate_sql":
                        generated_sql = state_update.get("sql_query", "")
                        sql_placeholder.markdown("**Generated SQL:**")
                        sql_placeholder.code(generated_sql, language="sql")
                    elif node_name == "respond":
                        final_response = state_update.get("final_answer", "")
                        response_placeholder.markdown(final_response)
        
        status_placeholder.success("✅ Analysis Complete")
        
        # Save to history 
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_response, 
            "sql": generated_sql
        })
