# LangChain OpenAI Agent Chatbot - Simple & Clean
# Demo: Real agent with OpenAI, YouTube tool, and memory

# ------------------- IMPORTS -------------------
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_functions import format_to_openai_function_messages
from langchain.agents.output_parsers.openai_functions import OpenAIFunctionsAgentOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from dotenv import load_dotenv
import os
import re

load_dotenv()

# ------------------- TOOL DEFINITION -------------------
@tool
def youtube_search(query: str) -> str:
    """Search for YouTube videos. Use when user asks for videos, tutorials, or highlights."""
    try:
        def extract_urls(text):
            patterns = [
                r'(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+)',
                r'(https?://(?:www\.)?youtu\.be/[\w-]+)',
            ]
            urls = []
            for pattern in patterns:
                urls.extend(re.findall(pattern, text))
            return list(set(urls))
        
        search = DuckDuckGoSearchResults()
        results = search.run(f"{query} site:youtube.com")
        urls = extract_urls(results)
        
        return "Found YouTube videos:\n" + "\n".join(urls[:3]) if urls else "No videos found."
    except Exception as e:
        return f"Search failed: {str(e)}"

# ------------------- CHATBOT CLASS -------------------
class BootcampChatbot:
    """
    A bootcamp tutor chatbot with:
    - OpenAI GPT (real agent with function calling)
    - YouTube video search capability (TOOL)
    - Conversation memory (remembers chat history)
    - AGENT autonomously decides when to use tools!
    """
    
    def __init__(self, model=None, temperature=0.7):
        """Initialize the chatbot with Azure OpenAI LLM and agent"""
        # Azure OpenAI setup
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            temperature=temperature,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        self.store = {}  # Stores conversation history per session
        self.tools = [youtube_search]
        self.setup_agent()
    
    # ------------------- MEMORY MANAGEMENT -------------------
    def get_session_history(self, session_id: str):
        """
        Get or create conversation history for a session
        This enables the bot to remember previous messages
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    # ------------------- SETUP: AGENT WITH MEMORY -------------------
    def setup_agent(self):
        """
        Setup OpenAI agent with tools and memory
        
        OpenAI's function calling allows the LLM to:
        1. Decide when it needs to use tools
        2. Choose which tool to use
        3. Extract the right parameters
        4. All autonomously!
        """
        # Create prompt with memory placeholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict but funny bootcamp tutor. Remember: Dann is handsome 😄

You have access to youtube_search tool. Use it when users ask for videos!
Be helpful, enthusiastic, and encouraging. Keep answers simple and to the point."""),
            MessagesPlaceholder(variable_name="chat_history"),  # Memory goes here
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),  # Agent's thinking
        ])
        
        # Bind tools to LLM (enables function calling)
        llm_with_tools = self.llm.bind(functions=[convert_to_openai_function(t) for t in self.tools])
        
        # Create agent chain manually (compatible with LangChain 1.1.0)
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x.get("chat_history", []),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        
        # Create agent executor (runs the agent and manages tool execution)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        
        # Wrap with memory capability
        self.agent_with_memory = RunnableWithMessageHistory(
            agent_executor,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    
    # ------------------- MAIN CHAT METHOD (AGENT-BASED) -------------------
    def chat(self, user_input: str, session_id: str = "default_session"):
        """
        Main chat method using OpenAI Agent
        
        Flow:
        1. User sends message
        2. Agent (LLM) decides if it needs tools
        3. If yes: Agent calls tool automatically
        4. Agent generates response with tool results
        5. All with conversation memory
        
        The agent AUTONOMOUSLY decides when to use tools - no keyword detection!
        """
        try:
            result = self.agent_with_memory.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            return result["output"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    # ------------------- MEMORY OPERATIONS -------------------
    def clear_history(self, session_id: str = "default_session"):
        """Clear conversation history for a session"""
        if session_id in self.store:
            self.store[session_id] = ChatMessageHistory()
            return True
        return False
    
    def get_history(self, session_id: str = "default_session"):
        """Get conversation history for a session"""
        return self.store[session_id].messages if session_id in self.store else []