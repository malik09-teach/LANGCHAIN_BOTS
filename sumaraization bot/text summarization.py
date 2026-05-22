import streamlit as st
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.tools import tool
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Helper Functions ---
def get_token_count(text: str, model_name: str = "gpt-4o") -> int:
    """Returns the exact number of tokens in a text string."""
    encoder = tiktoken.encoding_for_model(model_name)
    return len(encoder.encode(text))

def process_text(text: str) -> list[Document]:
    """Splits long text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# --- 2. Agent Factory ---
def initialize_agent(api_key: str):
    """Initializes the LLM, the dynamic tool, and the Agent."""
    llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=api_key)

    @tool
    def dynamic_summary_tool(text: str, preference: str = "fast") -> str:
        """
        Summarizes text automatically handling context limits.
        Args:
            text: The raw text to summarize.
            preference: Set to "detailed" ONLY if the user specifically asks for a thorough or highly detailed summary. Otherwise, leave as "fast".
        """
        TOKEN_LIMIT = 2040
        tokens = get_token_count(text)
        
        # We use st.info to show the backend routing decision in the UI
        st.info(f" **Token Count:** {tokens}")

        if tokens <= TOKEN_LIMIT:
            st.success(" **Strategy Selected:** `Stuff` (Context under limit)")
            docs = [Document(page_content=text)]
            chain = load_summarize_chain(llm, chain_type="stuff")
            return chain.invoke(docs)["output_text"]

        docs = process_text(text)

        if preference == "detailed":
            st.warning(" **Strategy Selected:** `Refine` (Long text + High Detail)")
            chain = load_summarize_chain(llm, chain_type="refine")
        else:
            st.warning(" **Strategy Selected:** `Map-Reduce` (Long text + Fast)")
            chain = load_summarize_chain(llm, chain_type="map_reduce")

        return chain.invoke(docs)["output_text"]

    tools = [dynamic_summary_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert summarization assistant. Use the dynamic_summary_tool to fulfill the user's request. Pay attention to whether they want a fast summary or a highly detailed one."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 3. Streamlit UI ---
st.set_page_config(page_title="Dynamic Agentic Summarizer", layout="wide")
st.title("Dynamic Summarization Agent ")
st.markdown("This agent automatically routes your text to `Stuff`, `Map-Reduce`, or `Refine` based on exact token counts and your preferred detail level.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("OpenAI API Key", type="password", help="Get this from platform.openai.com")
    
    st.markdown("---")
    st.header("Summary Style")
    style_choice = st.radio(
        "How detailed do you want the summary?",
        ["Standard & Fast", "Highly Detailed (Thorough)"],
        help="If the text is over 2040 tokens, 'Standard' uses Map-Reduce and 'Highly Detailed' uses Refine."
    )

# Main content area
input_text = st.text_area("Paste your content here:", height=300)

if st.button("Generate Summary", type="primary"):
    if not api_key_input:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not input_text.strip():
        st.error("Please paste some text to summarize.")
    else:
        with st.spinner("Agent is analyzing and summarizing..."):
            try:
                # 1. Initialize the agent with the user's API key
                agent_executor = initialize_agent(api_key=api_key_input)
                
                # 2. Formulate the prompt based on UI selection
                if style_choice == "Highly Detailed (Thorough)":
                    user_prompt = f"Please provide a highly detailed, comprehensive summary of the following text: {input_text}"
                else:
                    user_prompt = f"Please provide a quick, standard summary of the following text: {input_text}"
                
                # 3. Run the agent
                result = agent_executor.invoke({"input": user_prompt})
                
                # 4. Display results
                st.markdown("### Summary")
                st.write(result["output"])
                
            except Exception as e:
                st.error(f"An error occurred: {e}")