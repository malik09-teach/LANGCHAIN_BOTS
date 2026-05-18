# ================================
# IMPORT STREAMLIT
# ================================
# Streamlit is used for building the web UI
import streamlit as st


# ================================
# IMPORT GROQ LLM
# ================================
# ChatGroq connects LangChain with Groq models
from langchain_groq import ChatGroq


# ================================
# IMPORT SEARCH API WRAPPERS
# ================================
# These wrappers connect to external APIs
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper
)


# ================================
# IMPORT TOOLS
# ================================
# These are callable tools used by the AI agent
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)


# ================================
# IMPORT AGENT SYSTEM
# ================================
# initialize_agent creates AI agents
# AgentType defines reasoning behavior
from langchain.agents import initialize_agent, AgentType


# ================================
# IMPORT STREAMLIT CALLBACK
# ================================
# Shows live agent thinking process
from langchain.callbacks import StreamlitCallbackHandler


# ================================
# IMPORT PDF LOADER
# ================================
# Reads PDF documents
from langchain_community.document_loaders import PyPDFLoader


# ================================
# IMPORT TEXT SPLITTER
# ================================
# Splits large text into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ================================
# IMPORT EMBEDDINGS
# ================================
# Converts text into vector embeddings
from langchain_huggingface import HuggingFaceEmbeddings


# ================================
# IMPORT CHROMA VECTOR DATABASE
# ================================
# Stores vector embeddings
from langchain_chroma import Chroma


# ================================
# IMPORT RAG CHAIN
# ================================
# RetrievalQA is used for PDF question answering
from langchain.chains import RetrievalQA


# ================================
# IMPORT OS
# ================================
# Used for environment variables
import os


# ================================
# IMPORT DOTENV
# ================================
# Loads API keys from .env file
from dotenv import load_dotenv


# ================================
# LOAD .ENV FILE
# ================================
# Loads all environment variables
load_dotenv()


# ================================
# LOAD HUGGINGFACE TOKEN
# ================================
# Needed for embeddings model
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


# ================================
# CREATE EMBEDDING MODEL
# ================================
# Converts text into vectors
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# ==========================================================
# STREAMLIT UI
# ==========================================================

# App title
st.title("🔎 AI Search + PDF RAG Chatbot")


# App description
st.write(
    "Upload PDFs or search the web using AI tools."
)


# ==========================================================
# SIDEBAR SETTINGS
# ==========================================================

# Sidebar title
st.sidebar.title("Settings")


# Input field for Groq API Key
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password"
)


# ==========================================================
# CREATE SEARCH TOOLS
# ==========================================================

# Arxiv wrapper for research papers
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300
)


# Create Arxiv tool
arxiv = ArxivQueryRun(
    api_wrapper=arxiv_wrapper
)


# Wikipedia wrapper
wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300
)


# Create Wikipedia tool
wiki = WikipediaQueryRun(
    api_wrapper=wiki_wrapper
)


# DuckDuckGo internet search tool
search = DuckDuckGoSearchRun(
    name="Search"
)


# ==========================================================
# MULTIPLE PDF FILE UPLOADER
# ==========================================================

# Users can upload multiple PDFs
uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)


# ==========================================================
# CREATE CHAT MEMORY
# ==========================================================

# Checks whether messages exist in memory
if "messages" not in st.session_state:

    # Initial assistant message
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I can search the web "
                "and answer questions from PDFs."
            )
        }
    ]


# ==========================================================
# DISPLAY OLD CHAT HISTORY
# ==========================================================

# Loop through previous messages
for msg in st.session_state.messages:

    # Display each message
    st.chat_message(msg["role"]).write(
        msg["content"]
    )


# ==========================================================
# PDF PROCESSING
# ==========================================================

# Empty list for storing documents
documents = []


# Check whether user uploaded PDFs
if uploaded_files:

    # Loop through uploaded PDFs
    for uploaded_file in uploaded_files:

        # Create temporary PDF file
        with open("temp.pdf", "wb") as f:

            # Write uploaded PDF content
            f.write(uploaded_file.getvalue())


        # Load PDF file
        loader = PyPDFLoader("temp.pdf")


        # Extract PDF pages
        docs = loader.load()


        # Add pages into documents list
        documents.extend(docs)


    # ======================================================
    # TEXT SPLITTING
    # ======================================================

    # Split large text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


    # Create document chunks
    splits = text_splitter.split_documents(
        documents
    )


    # ======================================================
    # CREATE VECTOR DATABASE
    # ======================================================

    # Store chunks as embeddings
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )


    # Create retriever
    retriever = vectorstore.as_retriever()


# ==========================================================
# USER INPUT
# ==========================================================

# Chat input box
if prompt := st.chat_input(
    "Ask anything..."
):

    # ======================================================
    # SAVE USER MESSAGE
    # ======================================================

    # Store user message in memory
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # Display user message
    st.chat_message("user").write(prompt)


    # ======================================================
    # CREATE GROQ LLM
    # ======================================================

    # Create Groq language model
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192",
        streaming=True
    )


    # ======================================================
    # CREATE TOOLS LIST
    # ======================================================

    # List of tools available to the AI agent
    tools = [search, arxiv, wiki]


    # ======================================================
    # CREATE AI AGENT
    # ======================================================

    # AI agent capable of reasoning and tool calling
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True
    )


    # ======================================================
    # ASSISTANT RESPONSE AREA
    # ======================================================

    with st.chat_message("assistant"):

        # ==================================================
        # LIVE CALLBACK HANDLER
        # ==================================================

        # Shows live reasoning steps
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )


        # ==================================================
        # HYBRID LOGIC
        # ==================================================

        # If PDFs exist → use RAG
        if uploaded_files:

            # Create Retrieval QA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever
            )


            # Search PDFs and answer
            response = qa_chain.run(prompt)


        # Otherwise use web search agent
        else:

            # Agent searches web/tools
            response = search_agent.run(
                st.session_state.messages,
                callbacks=[st_cb]
            )


        # ==================================================
        # SAVE ASSISTANT RESPONSE
        # ==================================================

        # Store assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )


        # ==================================================
        # DISPLAY FINAL RESPONSE
        # ==================================================

        # Show final answer
        st.write(response)

