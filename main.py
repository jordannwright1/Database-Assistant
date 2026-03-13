import streamlit as st
import uuid
import os
os.environ["GOOGLE_AUTH_DISABLE_METADATA"] = "1"
os.environ["GCE_METADATA_HOST"] = "127.0.0.1"
from groq import Groq

# --- 1. Configuration & Setup ---
if "GROQ" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ"]["GROQ_API_KEY"]
else:
    st.error("GROQ API key not found in secrets under [GROQ] section!")

st.set_page_config(page_title="Austin Bikeshare AI", page_icon="🚲", layout="wide")
st.title("🚲 Austin Bikeshare AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- 2. Display History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sql"):
            st.code(message["sql"], language="sql")

# --- 3. Interaction Loop ---
if prompt := st.chat_input("Compare the average trip duration of 'Student Membership' users to 'Local365' users for the most recent month in the dataset. Additionally, for each of these two groups, calculate the percentage of their total trips that started during the morning rush hour (7 AM - 9 AM) on weekdays versus all other times."):
    from agent import app
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # UI Placeholders for Real-Time Updates
        status_text = st.empty()
        sql_placeholder = st.empty()
        response_placeholder = st.empty()
        
        inputs = {"question": prompt, "attempt": 0}
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        final_answer = ""
        generated_sql = ""

# --- UI Placeholders ---
        # Instead of st.empty(), use a dedicated log container
        log_container = st.container()
        sql_placeholder = st.empty()
        response_placeholder = st.empty()
        
        try:
            for output in app.stream(inputs, config=config):
                for node_name, state_update in output.items():
                    node_key = node_name.lower()
                    
                    # Log-style update: prevents messages from vanishing
                    if "plan" in node_key:
                        log_container.markdown("🧠 *Brainstorming query plan...*")
                    
                    elif "generate" in node_key or "sql" in node_key:
                        generated_sql = state_update.get("sql_query", "")
                        if generated_sql:
                            # Show "Working" message in the log
                            log_container.markdown("🏗️ *SQL Generated & Sanitized*")
                            sql_placeholder.code(generated_sql, language="sql")
                    
                    elif "respond" in node_key or "answer" in node_key:
                        final_answer = state_update.get("final_answer", "")
                        response_placeholder.markdown(final_answer)
            
            # Final success message at the bottom of the log
            log_container.success("✅ Analysis Complete")  
            
                      
            # Persist to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_answer, 
                "sql": generated_sql
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")
