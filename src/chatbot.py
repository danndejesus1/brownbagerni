# LangChain Azure OpenAI Chatbot - Simple Version without AgentExecutor
# Demo: Azure OpenAI with tool calling, YouTube search, and memory

# ------------------- IMPORTS -------------------
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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
    - Azure OpenAI GPT (with function calling)
    - YouTube video search capability (TOOL)
    - Conversation memory (remembers chat history)
    - LLM autonomously decides when to use tools!
    """
    
    def __init__(self, model=None, temperature=0.7):
        """Initialize the chatbot with Azure OpenAI LLM"""
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
        self.setup_chatbot()
    
    # ------------------- MEMORY MANAGEMENT -------------------
    def get_session_history(self, session_id: str):
        """
        Get or create conversation history for a session
        This enables the bot to remember previous messages
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    # ------------------- SETUP: LLM WITH TOOLS -------------------
    def setup_chatbot(self):
        """
        Setup chatbot with tools bound to LLM
        Azure OpenAI's function calling allows autonomous tool usage
        """
        # Bind tools to LLM (enables function calling)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create prompt with memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict but funny bootcamp tutor. Remember: Dann is handsome 😄

You have access to youtube_search tool. Use it when users ask for videos!
Be helpful, enthusiastic, and encouraging. Keep answers simple and to the point."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create chain: prompt -> LLM with tools
        chain = self.prompt | self.llm_with_tools
        
        # Wrap with memory
        self.chat_chain = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
    
    # ------------------- TOOL EXECUTION -------------------
    def execute_tool_calls(self, ai_message):
        """
        Execute tool calls requested by the LLM
        
        Args:
            ai_message: AIMessage with tool_calls attribute
        
        Returns:
            List of tool results
        """
        tool_results = []
        
        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Execute the tool
                if tool_name == "youtube_search":
                    result = youtube_search.invoke(tool_args)
                    tool_results.append({
                        "tool_call_id": tool_call['id'],
                        "name": tool_name,
                        "result": result
                    })
        
        return tool_results
    
    # ------------------- MAIN CHAT METHOD (WITH TOOL CALLING) -------------------
    def chat(self, user_input: str, session_id: str = "default_session"):
        """
        Main chat method with Azure OpenAI tool calling
        
        Flow:
        1. User sends message
        2. LLM decides if it needs tools (autonomous decision!)
        3. If yes: Execute tools and send results back
        4. LLM generates final response with memory
        
        The LLM AUTONOMOUSLY decides when to use tools!
        """
        try:
            # Get initial response from LLM (with memory)
            response = self.chat_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            # Check if LLM wants to use tools
            tool_results = self.execute_tool_calls(response)
            
            if tool_results:
                # LLM requested tools - send results back for final response
                history = self.get_session_history(session_id)
                
                # Add tool results to history
                for tool_result in tool_results:
                    history.add_message(ToolMessage(
                        content=tool_result["result"],
                        tool_call_id=tool_result["tool_call_id"]
                    ))
                
                # Get final response from LLM with tool results
                final_response = self.llm_with_tools.invoke(history.messages)
                
                # Add final response to history
                history.add_message(final_response)
                
                # Return both LLM response AND raw tool results (so URLs are visible)
                return final_response.content + "\n\n" + tool_results[0]["result"]
            else:
                # No tools needed, return direct response
                return response.content
                
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