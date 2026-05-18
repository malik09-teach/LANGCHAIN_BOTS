# ================================
# IMPORT STREAMLIT
# ================================
import streamlit as st

# ================================
# IMPORT GROQ LLM
# ================================
from langchain_groq import ChatGroq

# ================================
# IMPORT SEARCH API WRAPPERS & TOOLS
# ================================
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun

# ================================
# IMPORT MODERN AGENT SYSTEM
# ================================
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ================================
# IMPORT STREAMLIT CALLBACK
# ================================
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# ================================
# IMPORT PDF LOADER & TEXT SPLITTER
# ================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================================
# IMPORT EMBEDDINGS & VECTOR DB
# ================================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ================================
# IMPORT MODERN RAG CHAIN
# ================================
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain

# ================================
# IMPORT OS & DOTENV
# ================================
import os
from dotenv import load_dotenv

# ================================
# LOAD ENV VARIABLES
# ================================
load_dotenv()
os.environ["HUGGING_FACE_API"] = os.getenv("HUGGING_FACE_API")

# ================================
# CREATE EMBEDDING MODEL
# ================================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ==========================================================
# STREAMLIT UI
# ==========================================================
st.title("AI Search & PDF RAG Chatbot")
st.write("Upload PDFs or search the web using AI tools.")

# ==========================================================
# SIDEBAR SETTINGS
# ==========================================================
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# ==========================================================
# CREATE SEARCH TOOLS
# ==========================================================
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# ==========================================================
# MULTIPLE PDF FILE UPLOADER
# ==========================================================
uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

# ==========================================================
# CREATE CHAT MEMORY
# ==========================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can search the web and answer questions from PDFs."
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ==========================================================
# PDF PROCESSING
# ==========================================================
documents = []
retriever = None # Initialize retriever as None

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

# ==========================================================
# USER INPUT & LLM LOGIC
# ==========================================================
if user_query := st.chat_input("Ask anything..."):
    
    # Check for API key before proceeding
    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar.")
        st.stop()

    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Create Groq language model
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        streaming=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        final_answer = ""

        # ==================================================
        # HYBRID LOGIC
        # ==================================================
        # If PDFs exist → use Modern RAG
        if uploaded_files and retriever:
            
            # 1. Create a prompt for the QA system
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise.\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # 2. Build the retrieval chain
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            qa_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # 3. Invoke the chain
            response_dict = qa_chain.invoke({"input": user_query})
            final_answer = response_dict["answer"]

        # Otherwise use Modern Web Search Agent
        else:
            tools = [search, arxiv, wiki]
            
            # 1. Create an agent prompt template
            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful research assistant. Use the provided tools to answer the user's questions."),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            
            # 2. Build the tool calling agent and executor
            agent = create_tool_calling_agent(llm, tools, agent_prompt)
            search_agent = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
            
            # 3. Invoke the agent
            response_dict = search_agent.invoke(
                {"input": user_query},
                config={"callbacks": [st_cb]}
            )
            final_answer = response_dict["output"]

        # Display and save final response
        st.write(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})