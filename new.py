import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
from pandasai.responses.response_parser import ResponseParser
import os
from pathlib import Path
import re  # For parsing file paths in strings

# Initialize the language model with error handling
def initialize_llm():
    try:
        return ChatGroq(model_name="llama3-70b-8192", api_key='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')
    except KeyError:
        st.error("GROQ_API_KEY environment variable not set.")
        return None

llm = initialize_llm()

# Load data from a CSV file
def load_data(file_path):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return None
    else:
        st.error(f"File not found: {file_path}")
        return None

# Custom response parser
class CustomResponseParser(ResponseParser):
    def format_dataframe(self, result):
        return result["value"]

    def format_plot(self, result):
        if isinstance(result["value"], plt.Figure):
            output_dir = Path("exports/charts")
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / "latest_plot.png"
            try:
                result["value"].savefig(file_path)
                plt.close(result["value"])  # Close figure to prevent memory leaks
                return str(file_path)
            except Exception as e:
                st.error(f"Error saving plot: {e}")
                return "Error saving plot."
        return None

    def format_other(self, result):
        return result["value"]

# Process user queries with the dataframe
def chat_with_dataframe(query, df):
    if df is not None:
        try:
            query_engine = SmartDataframe(df, config={"llm": llm, "response_parser": CustomResponseParser})
            answer = query_engine.chat(query)
            return answer
        except Exception as e:
            return f"Error processing query: {e}"
    return "Dataframe is empty or not loaded."

# Callback function for handling query input
def handle_query():
    query = st.session_state["query_input"]
    if query:
        result = chat_with_dataframe(query, df)
        
        # Check if the result is a string containing an image path
        image_path_match = re.search(r"(?:[A-Z]:\\|/).+\.png", str(result))
        if image_path_match and os.path.exists(image_path_match.group(0)):
            st.session_state.chat_history.append({
                "user": query,
                "response": image_path_match.group(0),
                "type": "image"
            })
        elif isinstance(result, pd.DataFrame):  # DataFrame
            st.session_state.chat_history.append({
                "user": query,
                "response": result,
                "type": "dataframe"
            })
        elif isinstance(result, (int, float, str)):  # Scalar values or other text
            st.session_state.chat_history.append({
                "user": query,
                "response": result,
                "type": "text"
            })
        else:
            # Catch-all for unsupported response types
            st.session_state.chat_history.append({
                "user": query,
                "response": f"Unsupported response type: {type(result)}",
                "type": "error"
            })
        
        # Clear the input box dynamically
        st.session_state["query_input"] = ""

# Streamlit UI setup
st.set_page_config(page_title="Smart Dataframe Chat", page_icon="🤖", layout="wide")
st.title("📊 Smart Dataframe Chat with LLM Integration")
st.markdown("Ask questions about your dataset and get text, tables, or visualizations as responses.")

# Load the dataset
csv_file_path = "ai4i2020.csv"  # Replace with your CSV file path
df = load_data(csv_file_path)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if df is not None:
    st.success("CSV file loaded successfully!")
    st.dataframe(df.head())

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        if chat["type"] == "dataframe":
            st.markdown("**Response:**")
            st.dataframe(chat['response'])
        elif chat["type"] == "image":
            st.markdown("**Response:**")
            st.image(
                chat['response'], 
                caption="Generated Plot", 
                use_container_width=False,  # Set to False to customize the size
                width=500  # Adjust the width as needed (e.g., 500 pixels)
            )
        elif chat["type"] == "text":
            st.markdown(f"**Response:** {chat['response']}")
        elif chat["type"] == "error":
            st.error(f"**Response:** {chat['response']}")

    # Input box for user query with on_change callback
    st.text_input(
        "Ask your question:",
        key="query_input",
        on_change=handle_query
    )
else:
    st.warning("Please ensure the CSV file is in the correct path.")
