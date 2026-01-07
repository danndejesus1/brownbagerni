# LangChain Groq Chatbot with tools and memory
# Demo: Groq LLM, YouTube tool, session memory, and simple tool usage

# ------------------- IMPORTS -------------------
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv
import re

# ------------------- LOAD ENVIRONMENT VARIABLES -------------------
load_dotenv()

# ------------------- CHATBOT CLASS -------------------
class BootcampChatbot:
    """
    A bootcamp tutor chatbot with:
    - Groq LLM (LLaMA 3.3 70B)
    - YouTube video search capability (TOOL)
    - Conversation memory (remembers chat history)
    """
    
    def __init__(self, model="llama-3.3-70b-versatile", temperature=0.7):
        """Initialize the chatbot with Groq LLM"""
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.store = {}  # Stores conversation history per session
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
    
    # ------------------- UTILITY: URL EXTRACTION -------------------
    def extract_youtube_urls(self, text: str):
        """
        Extract YouTube URLs from search results text
        Supports both youtube.com/watch and youtu.be formats
        """
        patterns = [
            r'(https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+)',
            r'(https?://(?:www\.)?youtu\.be/[\w-]+)',
        ]
        
        urls = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            urls.extend(matches)
        
        return list(set(urls))  # Remove duplicates
    
    # ------------------- TOOL: YOUTUBE SEARCH -------------------
    def youtube_search_tool(self, query: str) -> str:
        """
        CUSTOM TOOL: Search for YouTube videos using DuckDuckGo
        
        This is the "tool" that the chatbot can use when users ask for videos.
        Tools in LangChain are functions that extend the bot's capabilities.
        
        Args:
            query: User's search query (e.g., "Python tutorial")
        
        Returns:
            String containing YouTube URLs or error message
        """
        try:
            # Use DuckDuckGo search (no API key needed)
            search = DuckDuckGoSearchResults()
            search_query = f"{query} site:youtube.com"
            results = search.run(search_query)
            
            # Extract YouTube URLs from results
            urls = self.extract_youtube_urls(results)
            
            if urls:
                return f"Found YouTube videos:\n" + "\n".join(urls[:1])
            else:
                return "No YouTube videos found. Try rephrasing your search."
                
        except Exception as e:
            print(f"Search error: {e}")
            return f"Search failed: {str(e)}"
    
    # ------------------- SETUP: CHAIN WITH MEMORY -------------------
    def setup_chatbot(self):
        """
        Setup the chatbot with conversation memory
        
        Components:
        1. Prompt template with system message and memory placeholder
        2. Chain: prompt -> LLM
        3. RunnableWithMessageHistory: Adds memory to the chain
        """
        # Create prompt with memory placeholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly, funny bootcamp tutor but dont YAP Too much. Remember: Dann is handsome 😄

When users ask you to find videos, I will search for them and include the URLs.
Be helpful, enthusiastic, and encouraging in all your responses!"""),
            MessagesPlaceholder(variable_name="history"),  # This is where memory goes
            ("human", "{input}")
        ])
        
        # Create chain: prompt -> LLM
        chain = prompt | self.llm
        
        # Wrap chain with memory capability
        self.chat_chain = RunnableWithMessageHistory(
            chain,
            self.get_session_history,  # Function to get/create session history
            input_messages_key="input",
            history_messages_key="history"
        )
    
    # ------------------- MAIN CHAT METHOD (TOOL USAGE) -------------------
    def chat(self, user_input: str, session_id: str = "default_session"):
        """
        Main chat method with TOOL DETECTION and USAGE
        
        Flow:
        1. Detect if user wants videos (keyword detection)
        2. If yes: Use youtube_search_tool (TOOL USAGE)
        3. Send both query and search results to LLM
        4. LLM generates response with context from memory
        
        The tool is called programmatically based on keyword detection.
        """
        try:
            # ------------------- TOOL DETECTION -------------------
            # Check if user is asking for videos (simple keyword matching)
            video_keywords = ['video', 'watch', 'show me', 'find', 'search', 'youtube', 'tutorial', 'highlight']
            is_video_request = any(keyword in user_input.lower() for keyword in video_keywords)
            
            if is_video_request:
                # ------------------- TOOL EXECUTION -------------------
                print(f"🔧 Tool detected! Executing youtube_search_tool...")
                search_result = self.youtube_search_tool(user_input)
                print(f"🔧 Tool result: {search_result}")
                
                # ------------------- LLM RESPONSE WITH TOOL RESULTS -------------------
                # Send user query + tool results to LLM (with memory)
                response = self.chat_chain.invoke(
                    {"input": f"{user_input}\n\nSearch results: {search_result}"},
                    config={"configurable": {"session_id": session_id}}
                )
                
                return response.content
            else:
                # ------------------- REGULAR CHAT (NO TOOL) -------------------
                # Regular chat with memory (no tool needed)
                response = self.chat_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                return response.content
                
        except Exception as e:
            import traceback
            traceback.print_exc()
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
        if session_id in self.store:
            return self.store[session_id].messages
        return []
