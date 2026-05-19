import os
import streamlit as st
from dotenv import load_dotenv 
from pathlib import Path 

# Modern LangChain Imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
import sqlite3

# Load environment variables
load_dotenv()

# --- 1. SET UP THE UI ---
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon='🦜')
st.title("🦜 LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

RADIO_OPT = ["Use SQLite Database (student.db)", "Connect to your MySQL Database"]

# Sidebar Configurations
st.sidebar.header("Database Configuration")
select_opt = st.sidebar.radio(label="Choose DB type", options=RADIO_OPT)

# Dynamic Input Fields based on DB selection
if RADIO_OPT.index(select_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host (e.g., localhost)")
    mysql_user = st.sidebar.text_input("MySQL User (e.g., root)")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database Name (e.g., university_db)")
else:
    db_uri = LOCALDB
    mysql_host, mysql_user, mysql_password, mysql_db = None, None, None, None

st.sidebar.header("LLM Configuration")
api_key = st.sidebar.text_input(label="Groq API Key", type="password")

# Validation Checks
if db_uri == MYSQL and not (mysql_host and mysql_user and mysql_password and mysql_db):
    st.info("Please enter all MySQL database connection details to continue.")
    st.stop()
    
if not api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()


# --- 2. INITIALIZE THE LLM ---
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile", # <-- The advanced reasoning model
    temperature=0, # <-- Set this to 0 for SQL tasks!
    streaming=True
)

# --- 3. CONFIGURE THE DATABASE CONNECTION ---
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        # Fallback to local SQLite file
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        # Connect to live MySQL server
        connection_string = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        return SQLDatabase(create_engine(connection_string))

db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)


# --- 4. BUILD THE AGENT ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="zero-shot-react-description"
)


# --- 5. MANAGE CHAT HISTORY ---
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you query the database today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# --- 6. HANDLE USER QUERY ---
user_query = st.chat_input(placeholder="Ask anything from the database...")

if user_query:
    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Generate and stream agent response
    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        
        try:
            # Execute the agent and capture the final answer
            response = agent.run(user_query, callbacks=[streamlit_callback])
            
            # Save and display the final answer
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")