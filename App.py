from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain API setup
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A ChatBot with Ollama"

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, engine, temperature, max_tokens):
    try:
        # Initialize LLM
        llm = ollama.Ollama(model=engine)
        
        # Parse response
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({"question": question})
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar options for model and parameters
engine = st.sidebar.selectbox("Select a Model", ["llama3", "gemma:2b"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.title("Q&A ChatBot with Ollama")
st.write("Ask any question, and I will try to help!")

user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    response = generate_response(user_input, engine, temperature, max_tokens)
    st.write("Assistant:", response)
else:
    st.write("Please provide a question to get started.")


