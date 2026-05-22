import streamlit as st
import tiktoken
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader

# --- 1. Helper Functions ---
def get_token_count(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def process_text(text: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=400)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

def fetch_youtube_transcript(url: str) -> str:
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    docs = loader.load()
    return " ".join([d.page_content for d in docs])

def fetch_website_text(url: str) -> str:
    loader = WebBaseLoader(url)
    docs = loader.load()
    return " ".join([d.page_content for d in docs])

# --- 2. The Optimized Router (No Agent Overhead) ---
def execute_fast_summary(api_key: str, text: str, detail_level: str):
    """Routes text based on length and uses a dual-model architecture for speed."""
    
    # Model 1: Fast 8B model for mapping (heavy lifting)
    fast_llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=api_key)
    
    # Model 2: Smart 70B model for reducing (final polishing)
    smart_llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)

    tokens = get_token_count(text)
    st.info(f"📊 **Approximate Token Count:** {tokens}")

    # PATH A: Short Text
    if tokens <= 2040:
        st.success("⚡ **Strategy Selected:** `Fast Stuff`")
        
        prompt_template = "Summarize the following text:\n\n{text}" if detail_level == "Standard & Fast" else "Provide a highly detailed, comprehensive summary of the following text, capturing all core nuances:\n\n{text}"
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        chain = load_summarize_chain(smart_llm, chain_type="stuff", prompt=prompt)
        return chain.invoke([Document(page_content=text)])["output_text"]

    # PATH B: Long Text
    st.warning("🚀 **Strategy Selected:** `Tiered Map-Reduce` (Parallel)")
    docs = process_text(text)
    
    map_prompt = PromptTemplate(template="Summarize this section of text:\n\n{text}", input_variables=["text"])
    reduce_template = "Combine the following summaries into a cohesive final summary:\n\n{text}" if detail_level == "Standard & Fast" else "Synthesize the following summaries into a highly detailed, comprehensive master summary. Ensure no key information is lost:\n\n{text}"
    reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["text"])

    # We pass the fast LLM to the entire chain to maximize Groq's parallel processing speed
    chain = load_summarize_chain(
        llm=fast_llm, 
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt
    )

    return chain.invoke(docs)["output_text"]

# --- 3. Streamlit UI ---
st.set_page_config(page_title="Omni-Summarizer Pro", layout="wide")
st.title("Omni-Source Summarization App ⚡")

with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Groq API Key", type="password")
    st.markdown("---")
    style_choice = st.radio(
        "Summary Style:",
        ["Standard & Fast", "Highly Detailed (Thorough)"]
    )

tab1, tab2, tab3 = st.tabs(["📝 Paste Text", "🎥 YouTube Video", "🔗 Website"])

with tab1:
    input_text = st.text_area("Paste your content here:", height=200)
    process_text_btn = st.button("Summarize Text", type="primary")

with tab2:
    yt_url = st.text_input("Paste YouTube URL:")
    process_yt_btn = st.button("Summarize Video", type="primary")

with tab3:
    web_url = st.text_input("Paste Website URL:")
    process_web_btn = st.button("Summarize Website", type="primary")

# Execute Logic
if process_text_btn or process_yt_btn or process_web_btn:
    if not api_key_input:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        raw_content = ""
        try:
            with st.spinner("Fetching content..."):
                if process_text_btn:
                    if not input_text.strip():
                        st.warning("Please paste some text first.")
                        st.stop()
                    raw_content = input_text
                    
                elif process_yt_btn:
                    if not yt_url.strip():
                        st.warning("Please provide a YouTube URL.")
                        st.stop()
                    raw_content = fetch_youtube_transcript(yt_url)
                    
                elif process_web_btn:
                    if not web_url.strip():
                        st.warning("Please provide a Website URL.")
                        st.stop()
                    raw_content = fetch_website_text(web_url)

            with st.spinner("AI is analyzing and summarizing..."):
                summary = execute_fast_summary(
                    api_key=api_key_input, 
                    text=raw_content, 
                    detail_level=style_choice
                )
                
                st.markdown("### Final Summary")
                st.write(summary)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")