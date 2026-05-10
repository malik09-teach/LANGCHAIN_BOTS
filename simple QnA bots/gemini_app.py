import os
from dotenv import load_dotenv
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "sprint_65hrs"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant and respond in stepwise detail."),
        ("user", "{question}")
    ]
)

def generate_response(question, llm_model, temperature, max_tokens, apikey):

    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=apikey
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    answer = chain.invoke({"question": question})

    return answer


st.title("Google Q&A BOT WITH LANGSMITH")

st.sidebar.title("Settings")

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

apikey = st.sidebar.text_input(
    "Enter your Google API Key:",
    type="password"
)

llm = st.sidebar.selectbox(
    "Select Gemini Model",
    ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]
)

max_tokens = st.sidebar.slider(
    "Tokens",
    min_value=20,
    max_value=5000,
    value=500
)

st.write("Ask a question from the bot")

user_input = st.text_input("You")

if user_input and apikey:

    response = generate_response(
        user_input,
        llm,
        temperature,
        max_tokens,
        apikey
    )

    st.write(response)

elif user_input and not apikey:
    st.warning("Please enter your Google API Key")

else:
    st.write("Please provide user input")