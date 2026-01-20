
# Minimal Streamlit LLM Chatbot (No tools, memory, or agent)

import streamlit as st
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="LLM Only Chatbot", layout="centered")
st.title("Brown Bag: LLM Only Chatbot")
st.caption("No tools, no memory, no agent. Just LLM!")

@st.cache_resource
def get_llm():
	return AzureChatOpenAI(
		azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
		api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
		temperature=0.7,
		azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
		api_key=os.getenv("AZURE_OPENAI_API_KEY")
	)

llm = get_llm()

if "messages" not in st.session_state:
	st.session_state.messages = []


for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything (no tools, no memory)..."):
	st.session_state.messages.append({"role": "user", "content": prompt})
	with st.chat_message("user"):
		st.markdown(prompt)
	with st.chat_message("assistant"):
		with st.spinner("Thinking..."):
			try:
				
				response = llm.invoke(prompt)
				st.markdown(response.content)
				st.session_state.messages.append({"role": "assistant", "content": response.content})
			except Exception as e:
				st.error(f"Error: {str(e)}")
				st.info("Check your Azure OpenAI API key and endpoint in your .env file!")

st.divider()
st.caption("Built with Streamlit & LangChain (LLM only)")
