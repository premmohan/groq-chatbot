import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# 1. Load environment variables
load_dotenv()

# 2. Configure Streamlit Page
st.set_page_config(page_title="Groq Stateful Chatbot", page_icon="🤖")
st.title("🤖 Groq Chatbot")
st.caption("A chatbot that remembers context and trims old messages.")

# 3. Initialize the Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 4. Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Define Trimming Logic
def trim_history(history, max_messages=6):
    """Keep only the most recent N messages to save tokens and maintain speed."""
    if len(history) > max_messages:
        return history[-max_messages:]
    return history

# 6. Create Chat Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides concise and accurate answers."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 7. Display Chat History from Session State
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# 8. Handle User Input
if user_query := st.chat_input("Type your message here..."):
    
    # Display user message in UI
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate AI Response
    with st.chat_message("assistant"):
        # a. Trim history before sending to LLM
        trimmed_history = trim_history(st.session_state.messages)
        
        # b. Format prompt with history and new input
        formatted_prompt = prompt_template.invoke({
            "chat_history": trimmed_history,
            "input": user_query
        })
        
        # c. Call LLM
        response = llm.invoke(formatted_prompt)
        ai_answer = response.content
        st.markdown(ai_answer)

    # 9. Update Session State (Maintain History)
    st.session_state.messages.append(HumanMessage(content=user_query))
    st.session_state.messages.append(AIMessage(content=ai_answer))

# Sidebar Info
with st.sidebar:
    st.header("Settings")
    st.info("This chatbot uses **llama-3.1-8b-instant** built by Prem Mohan")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()