# ------------------- IMPORTS -------------------
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

load_dotenv()

# ------------------- TOOL DEFINITION -------------------
@tool
def youtube_search(query: str, max_results: int = 1) -> str:
    try:
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            return "YouTube API key not configured. Please set YOUTUBE_API_KEY in .env"
        
        # Limit max_results
        max_results = max(1, min(max_results, 10))
        
        youtube = build("youtube", "v3", developerKey=api_key)
        
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results 
        )
        response = request.execute()
        
        if not response.get("items"):
            return "No videos found."
        
        results = []
        for item in response["items"]:
            title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            url = f"https://www.youtube.com/watch?v={video_id}"
            results.append(f"• {title}\n  {url}")
        
        return f"Found {len(results)} YouTube videos:\n" + "\n\n".join(results)
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
    
    # ------------------- SETUP: AGENT WITH LANGGRAPH -------------------
    def setup_chatbot(self):
        """
        Setup chatbot with LangGraph's ReAct Agent
        The Agent handles tool calling autonomously in a loop!
        """
        # System prompt for the agent
        system_prompt = """You are a strict but funny bootcamp tutor. Remember: Dann is  😄

You have access to youtube_search tool. Use it when users ask for videos!
Be helpful, enthusiastic, and encouraging. Keep answers simple and to the point."""
        
        # Create the ReAct agent using LangGraph (handles tool calling automatically)
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=system_prompt
        )
    
    # ------------------- MAIN CHAT METHOD (AGENT HANDLES TOOLS) -------------------
    def chat(self, user_input: str, session_id: str = "default_session"):
      
        try:
            # Get conversation history
            history = self.get_session_history(session_id)
            
            # Build messages: history + new user input
            messages = list(history.messages) + [HumanMessage(content=user_input)]
            
            # Agent handles everything: tool decisions, execution, and final response
            response = self.agent.invoke({"messages": messages})
            
            # Get the final AI message
            final_message = response["messages"][-1]
            
            # Save to history
            history.add_user_message(user_input)
            history.add_ai_message(final_message.content)
            
            return final_message.content
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    # ------------------- MEMORY OPERATIONS -------------------
    def clear_history(self, session_id: str = "default_session"):
     
        if session_id in self.store:
            self.store[session_id] = ChatMessageHistory()
            return True
        return False
    
    def get_history(self, session_id: str = "default_session"):
      
        return self.store[session_id].messages if session_id in self.store else []