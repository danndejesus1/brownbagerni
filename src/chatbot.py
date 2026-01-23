# ------------------- IMPORTS -------------------
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
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
    """Search for YouTube videos when user asks for videos, tutorials, or highlights."""
    try:
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            return "YouTube API key not configured."
        
        max_results = max(1, min(max_results, 5))
        youtube = build("youtube", "v3", developerKey=api_key)
        
        response = youtube.search().list(
            part="snippet", q=query, type="video", maxResults=max_results
        ).execute()
        
        if not response.get("items"):
            return "No videos found."
        
        results = []
        for item in response["items"]:
            title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            results.append(f"- {title}\n  https://www.youtube.com/watch?v={video_id}")
        
        return f"Found {len(results)} videos:\n" + "\n\n".join(results)
    except Exception as e:
        return f"Search failed: {str(e)}"


# ------------------- CHATBOT CLASS -------------------
class BootcampChatbot:
    """LangGraph agent chatbot with YouTube search and live visualization."""
    
    # ------------------- INITIALIZATION -------------------
    def __init__(self, temperature=0.7):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            temperature=temperature,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        self.store = {}
        self.tools = [youtube_search]
        
        # Create the ReAct agent with LangGraph
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt="""You are a strict but funny bootcamp tutor.
You have access to youtube_search tool. Use it when users ask for videos!
Be helpful and keep answers simple. Also Remember that dann is handsome!"""
        )
    
    # ------------------- MEMORY MANAGEMENT -------------------
    def get_session_history(self, session_id: str):
        """Get or create conversation history for a session."""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def clear_history(self, session_id: str = "default_session"):
        """Clear conversation history for a session."""
        if session_id in self.store:
            self.store[session_id] = ChatMessageHistory()
            return True
        return False
    
    # ------------------- CHAT METHODS -------------------
    def chat(self, user_input: str, session_id: str = "default_session"):
        """Simple chat method - returns response only."""
        response, _ = self.chat_with_trace(user_input, session_id)
        return response
    
    def chat_with_trace(self, user_input: str, session_id: str = "default_session"):
        """Chat with trace for visualization. Returns: (response, trace_steps)"""
        try:
            history = self.get_session_history(session_id)
            messages = list(history.messages) + [HumanMessage(content=user_input)]
            
            response = self.agent.invoke({"messages": messages})
            new_messages = response["messages"][len(messages) - 1:]
            trace_steps = self._extract_trace(new_messages)
            
            final_message = response["messages"][-1]
            history.add_user_message(user_input)
            history.add_ai_message(final_message.content)
            
            return final_message.content, trace_steps
        except Exception as e:
            return f"Error: {str(e)}", []
    
    # ------------------- LIVE STREAMING WITH TRACE -------------------
    def chat_with_live_trace(self, user_input: str, session_id: str = "default_session"):
        """
        Generator that yields live graph updates as agent executes.
        Yields: (current_node, used_tools, is_complete, message, trace)
        
        This enables real-time visualization of the agent's decision flow!
        """
        try:
            history = self.get_session_history(session_id)
            messages = list(history.messages) + [HumanMessage(content=user_input)]
            
            # Starting node
            yield ("start", False, False, None)
            
            used_tools = False
            for chunk in self.agent.stream({"messages": messages}):
                if "agent" in chunk:
                    # Agent is processing
                    yield ("agent", used_tools, False, None)
                    agent_msg = chunk["agent"]["messages"][-1]
                    if hasattr(agent_msg, "tool_calls") and agent_msg.tool_calls:
                        used_tools = True
                        yield ("agent_to_tools", used_tools, False, None)
                elif "tools" in chunk:
                    # Tools are executing
                    yield ("tools", used_tools, False, None)
                    yield ("tools_to_agent", used_tools, False, None)
            
            # Get final response
            response = self.agent.invoke({"messages": messages})
            final_message = response["messages"][-1].content
            
            # Save to history
            history.add_user_message(user_input)
            history.add_ai_message(final_message)
            
            new_messages = response["messages"][len(messages) - 1:]
            trace_steps = self._extract_trace(new_messages)
            
            # Complete
            yield ("end", used_tools, True, final_message, trace_steps)
        except Exception as e:
            yield ("error", False, True, f"Error: {str(e)}", [])
    
    # ------------------- TRACE EXTRACTION -------------------
    def _extract_trace(self, messages):
        """Extract trace from messages for storage."""
        trace = []
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            if msg_type == "HumanMessage":
                trace.append({"step": i+1, "node": "User Input", "action": "receive", "details": msg.content[:100]})
            elif msg_type == "AIMessage":
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        trace.append({"step": i+1, "node": "Agent Decision", "action": "call_tool", "details": f"Tool: {tc.get('name')}"})
                else:
                    trace.append({"step": i+1, "node": "Agent Response", "action": "generate", "details": msg.content[:100]})
            elif msg_type == "ToolMessage":
                trace.append({"step": i+1, "node": "Tool Execution", "action": "execute", "details": msg.content[:80]})
        return trace
    
    # ------------------- GRAPH VISUALIZATION (LIVE) -------------------
    def get_live_graph_dot(self, current_node: str, used_tools: bool):
        """
        Generate live DOT graph with current node highlighted.
        Colors: Green = done, Blue = current, Gray = inactive
        """
        ACTIVE, CURRENT, INACTIVE, WAITING = "#10b981", "#3b82f6", "#374151", "#6b7280"
        
        # Determine node states based on current position
        states = {"start": "done", "agent": "inactive", "tools": "inactive", "end": "inactive"}
        
        if current_node == "start": 
            states["start"] = "current"
        elif current_node == "agent": 
            states["start"], states["agent"] = "done", "current"
        elif current_node == "agent_to_tools": 
            states["start"], states["agent"], states["tools"] = "done", "done", "waiting"
        elif current_node == "tools": 
            states["start"], states["agent"], states["tools"] = "done", "done", "current"
        elif current_node == "tools_to_agent": 
            states["start"], states["agent"], states["tools"] = "done", "current", "done"
        elif current_node == "end":
            states = {"start": "done", "agent": "done", "tools": "done" if used_tools else "inactive", "end": "current"}
        
        def color(s): 
            if s == "current": return CURRENT
            elif s == "done": return ACTIVE
            elif s == "waiting": return WAITING
            return INACTIVE
        
        # Build DOT graph
        lines = [
            'digraph G { bgcolor="#0f172a"; rankdir=TB; splines=ortho; nodesep=0.6; ranksep=0.8;',
            'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, fontcolor="#fff", margin="0.25,0.15", width=1.5];',
            'edge [penwidth=2.5, arrowsize=0.8];'
        ]
        
        # Add nodes
        for node, label in [("start", "__start__"), ("agent", "agent"), ("tools", "tools"), ("end", "__end__")]:
            c = color(states[node])
            border = "#60a5fa" if states[node] == "current" else c
            pw = 3 if states[node] == "current" else 1
            lines.append(f'{node} [label="{label}", fillcolor="{c}", color="{border}", penwidth={pw}];')
        
        # Add edges
        ea, ec, ei = ACTIVE, CURRENT, "#4b5563"
        lines.append(f'start -> agent [color="{ea if states["agent"] != "inactive" else ei}"];')
        lines.append(f'agent -> tools [color="{ec if current_node == "agent_to_tools" else ea if states["tools"] in ["done","current"] else ei}"];')
        lines.append(f'tools -> agent [color="{ec if current_node == "tools_to_agent" else ea if used_tools and states["end"] != "inactive" else ei}"];')
        lines.append(f'agent -> end [color="{ea if states["end"] != "inactive" else ei}"];')
        lines.append("}")
        
        return "\n".join(lines)
    
 