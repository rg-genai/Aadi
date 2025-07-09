import streamlit as st
import pandas as pd
import json
import re
import google.generativeai as genai

# --- Configuration ---
st.set_page_config(page_title="Project Aadi Conversational Agent", layout="wide")

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Error configuring Google AI API. Please check your 'secrets.toml' file. Details: {e}")
    st.stop()

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if 'UserID' in df.columns:
            df.set_index('UserID', inplace=True)
            return df
        else:
            st.error("The JSON data must contain a 'UserID' field for each record.")
            return None
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return None

# --- Helper Functions ---
def find_user_id(query):
    match = re.search(r'AADI-\d+', query.upper())
    if match:
        return match.group(0)
    return None

def get_user_data(df, user_id):
    try:
        return df.loc[user_id].to_dict()
    except KeyError:
        return None

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Return a user-friendly error message
        st.error(f"An error occurred while communicating with the Gemini API: {e}")
        return None

# --- Main Application ---
st.title("💬 Project Aadi: Your Conversational Financial Co-Pilot")
st.caption("Now with memory and holistic reasoning capabilities.")

# Load the financial data
financial_df = load_data('financial_data.json')

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your finances today? Please include a UserID (e.g., AADI-001) in your question."}]

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The new chat input widget
if prompt := st.chat_input("Ask any question about your finances..."):
    # Add user's message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- The New "Brain" Logic ---
    with st.chat_message("assistant"):
        with st.spinner("Aadi is thinking..."):
            user_id = find_user_id(prompt)
            
            if not user_id and "last_user_id" in st.session_state:
                user_id = st.session_state.last_user_id
            
            if user_id:
                # Store the last used user_id for follow-up questions
                st.session_state.last_user_id = user_id
                
                user_data = get_user_data(financial_df, user_id)
                if user_data:
                    # Construct a single, powerful prompt for the LLM
                    conversation_history = "\n".join([f'{m["role"]}: {m["content"]}' for m in st.session_state.messages])
                    
                    master_prompt = f"""
                    You are 'Aadi', an expert AI financial co-pilot. Your primary goal is to provide comprehensive, well-researched financial advice by analyzing the user's complete financial data.

                    **Your Task:**
                    1.  Analyze the user's latest question in the context of the entire conversation history.
                    2.  Thoroughly examine ALL relevant sections of the provided JSON data (User data, Credit report, EPF, Net worth, and Transactions) to formulate your answer. Do not limit your analysis to just one section.
                    3.  If the user's question is ambitious or vague (e.g., "How can I retire at 40?"), first ask clarifying questions to understand their goals (e.g., "That's a great goal! To help you, could you tell me what kind of lifestyle you envision in retirement?"). Then, wait for their response before providing a detailed plan.
                    4.  Provide your answer in a clear, conversational, and helpful tone.

                    **Conversation History:**
                    {conversation_history}

                    **Complete Financial Data for {user_id}:**
                    ```json
                    {json.dumps(user_data, indent=2)}
                    ```

                    **User's Latest Question:** "{prompt}"

                    Please provide your expert response now. If the question is about retirement or a complex goal, ask a clarifying question first.
                    """
                    
                    response = get_gemini_response(master_prompt)
                    if response:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error(f"Data for UserID {user_id} could not be found.")
            else:
                response = "I can't seem to find a UserID in your question. Please include a UserID (like AADI-001) so I can access the correct financial data."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})