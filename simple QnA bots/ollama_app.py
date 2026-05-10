import os
from dotenv import load_dotenv
import streamlit as st

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant and respond in stepwise detail."),
        ("user", "{question}")
    ]
)

def generate_response(question, llm_model, temperature, max_tokens):

    llm = OllamaLLM(
        model=llm_model,
        temperature=temperature,
        num_predict=max_tokens
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    answer = chain.invoke({"question": question})

    return answer


st.title("Q&A BOT WITH LANGSMITH")

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

llm = st.sidebar.selectbox(
    "Select Open Source Model",
    ["gemma2:2b"]
)

max_tokens = st.sidebar.slider(
    "Tokens",
    min_value=20,
    max_value=5000,
    value=500
)

st.write("Ask a question from the bot")

user_input = st.text_input("You")

if user_input:

    response = generate_response(
        user_input,
        llm,
        temperature,
        max_tokens
    )

    st.write(response)

else:
    st.write("Please provide user input")