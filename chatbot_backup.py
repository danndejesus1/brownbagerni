# LangChain Azure OpenAI Chatbot - Agent Version with LangGraph
# Demo: Azure OpenAI with Agent, YouTube search, and memory

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
    """Search for YouTube videos. Use when user asks for videos, tutorials, or highlights.
    
    Args:
        query: Search query for YouTube
        max_results: Number of videos to return (default: 1, max: 10)
    """
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
        system_prompt = """You are a strict but funny bootcamp tutor. Remember: Dann is handsome 😄

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
        """
        Main chat method using LangGraph ReAct Agent
        
        Flow:
        1. User sends message
        2. Agent decides if it needs tools (autonomous decision!)
        3. Agent executes tools in a loop if needed
        4. Agent generates final response
        
        The Agent handles the entire tool calling loop!
        """
        response, _ = self.chat_with_trace(user_input, session_id)
        return response
    
    def chat_with_trace(self, user_input: str, session_id: str = "default_session"):
        """
        Chat method that also returns agent decision trace for visualization.
        Returns: (response_text, trace_steps)
        """
        try:
            # Get conversation history
            history = self.get_session_history(session_id)
            
            # Build messages: history + new user input
            messages = list(history.messages) + [HumanMessage(content=user_input)]
            
            # Agent handles everything: tool decisions, execution, and final response
            response = self.agent.invoke({"messages": messages})
            
            # Extract trace/steps from ONLY the new messages (current turn)
            # Skip the history messages, only trace the new ones
            num_input_messages = len(messages)
            new_messages = response["messages"][num_input_messages - 1:]  # Include user input + new
            trace_steps = self._extract_trace(new_messages)
            
            # Get the final AI message
            final_message = response["messages"][-1]
            
            # Save to history
            history.add_user_message(user_input)
            history.add_ai_message(final_message.content)
            
            return final_message.content, trace_steps
                
        except Exception as e:
            return f"Error: {str(e)}", []
    
    def chat_with_live_trace(self, user_input: str, session_id: str = "default_session"):
        """
        Generator that yields live graph updates as agent executes.
        Yields: (current_node, used_tools, is_complete, final_message)
        """
        try:
            history = self.get_session_history(session_id)
            messages = list(history.messages) + [HumanMessage(content=user_input)]
            
            # Yield: starting
            yield ("start", False, False, None)
            
            used_tools = False
            final_message = None
            
            # Stream the agent execution
            for chunk in self.agent.stream({"messages": messages}):
                if "agent" in chunk:
                    # Agent is thinking
                    yield ("agent", used_tools, False, None)
                    
                    # Check if agent decided to use tools
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
            
            # Extract trace for storage
            num_input_messages = len(messages)
            new_messages = response["messages"][num_input_messages - 1:]
            trace_steps = self._extract_trace(new_messages)
            
            # Yield: complete
            yield ("end", used_tools, True, final_message, trace_steps)
                
        except Exception as e:
            yield ("error", False, True, f"Error: {str(e)}", [])
    
    def get_live_graph_dot(self, current_node: str, used_tools: bool):
        """
        Generate animated LangGraph-style DOT visualization.
        Highlights the current node being executed.
        """
        # Colors - clean, professional palette
        active_color = "#10b981"      # Green - completed
        current_color = "#3b82f6"     # Blue - currently executing
        inactive_color = "#374151"    # Dark gray - not yet executed
        waiting_color = "#6b7280"     # Medium gray - queued
        
        # Determine node states based on current position
        node_states = {
            "start": "done",
            "agent": "inactive",
            "tools": "inactive", 
            "end": "inactive"
        }
        
        if current_node == "start":
            node_states["start"] = "current"
        elif current_node == "agent":
            node_states["start"] = "done"
            node_states["agent"] = "current"
        elif current_node == "agent_to_tools":
            node_states["start"] = "done"
            node_states["agent"] = "done"
            node_states["tools"] = "waiting"
        elif current_node == "tools":
            node_states["start"] = "done"
            node_states["agent"] = "done"
            node_states["tools"] = "current"
        elif current_node == "tools_to_agent":
            node_states["start"] = "done"
            node_states["agent"] = "current"
            node_states["tools"] = "done"
        elif current_node == "end":
            node_states["start"] = "done"
            node_states["agent"] = "done"
            node_states["tools"] = "done" if used_tools else "inactive"
            node_states["end"] = "current"
        
        def get_color(state):
            if state == "current":
                return current_color
            elif state == "done":
                return active_color
            elif state == "waiting":
                return waiting_color
            return inactive_color
        
        def get_font(state):
            return "#ffffff"
        
        dot_lines = [
            "digraph LangGraph {",
            '    bgcolor="#0f172a";',
            '    rankdir=TB;',
            '    splines=ortho;',
            '    nodesep=0.6;',
            '    ranksep=0.8;',
            '    node [shape=box, style="rounded,filled", fontname="SF Pro Display, Helvetica, Arial", fontsize=13, fontcolor="#ffffff", margin="0.25,0.15", width=1.5];',
            '    edge [fontname="Helvetica", penwidth=2.5, arrowsize=0.8];',
            '',
        ]
        
        # Nodes with cleaner styling
        node_labels = [("start", "__start__"), ("agent", "agent"), ("tools", "tools"), ("end", "__end__")]
        
        for node, label in node_labels:
            color = get_color(node_states[node])
            font = get_font(node_states[node])
            
            if node_states[node] == "current":
                # Highlighted current node with glow effect
                dot_lines.append(f'    {node} [label="{label}", fillcolor="{color}", color="#60a5fa", fontcolor="{font}", penwidth=3];')
            else:
                dot_lines.append(f'    {node} [label="{label}", fillcolor="{color}", color="{color}", fontcolor="{font}"];')
        
        # Edges with smooth transitions
        edge_active = "#10b981"
        edge_current = "#3b82f6"
        edge_inactive = "#4b5563"
        
        # start -> agent
        e_color = edge_active if node_states["agent"] != "inactive" else edge_inactive
        dot_lines.append(f'    start -> agent [color="{e_color}"];')
        
        # agent -> tools
        if current_node == "agent_to_tools":
            e_color = edge_current
        elif node_states["tools"] in ["done", "current"]:
            e_color = edge_active
        else:
            e_color = edge_inactive
        dot_lines.append(f'    agent -> tools [color="{e_color}"];')
        
        # tools -> agent
        if current_node == "tools_to_agent":
            e_color = edge_current
        elif used_tools and node_states["end"] != "inactive":
            e_color = edge_active
        else:
            e_color = edge_inactive
        dot_lines.append(f'    tools -> agent [color="{e_color}"];')
        
        # agent -> end
        e_color = edge_active if node_states["end"] != "inactive" else edge_inactive
        dot_lines.append(f'    agent -> end [color="{e_color}"];')
        
        dot_lines.append("}")
        return "\n".join(dot_lines)
    
    def _extract_trace(self, messages):
        """
        Extract agent decision trace from messages for visualization.
        Returns a list of step dicts: {node, action, details}
        """
        trace = []
        step_num = 0
        
        for msg in messages:
            msg_type = type(msg).__name__
            
            if msg_type == "HumanMessage":
                step_num += 1
                trace.append({
                    "step": step_num,
                    "node": "User Input",
                    "action": "receive",
                    "details": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                })
            
            elif msg_type == "AIMessage":
                # Check if AI decided to call tools
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        step_num += 1
                        trace.append({
                            "step": step_num,
                            "node": "Agent Decision",
                            "action": "call_tool",
                            "details": f"Tool: {tool_call.get('name', 'unknown')}"
                        })
                else:
                    step_num += 1
                    trace.append({
                        "step": step_num,
                        "node": "Agent Response",
                        "action": "generate",
                        "details": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    })
            
            elif msg_type == "ToolMessage":
                step_num += 1
                trace.append({
                    "step": step_num,
                    "node": "Tool Execution",
                    "action": "execute",
                    "details": f"Result: {msg.content[:80]}..." if len(msg.content) > 80 else f"Result: {msg.content}"
                })
        
        return trace
    
    def get_graph_dot(self, trace_steps):
        """
        Generate a LangGraph-style DOT visualization.
        Shows the agent graph structure with highlighted execution path.
        """
        if not trace_steps:
            return None
        
        # Determine which path was taken
        used_tools = any(step["node"] in ["Agent Decision", "Tool Execution"] for step in trace_steps)
        
        # Colors - matching live graph
        active_color = "#10b981"      # Green - completed
        inactive_color = "#374151"    # Dark gray
        edge_active = "#10b981"
        edge_inactive = "#4b5563"
        
        dot_lines = [
            "digraph LangGraph {",
            '    bgcolor="#0f172a";',
            '    rankdir=TB;',
            '    splines=ortho;',
            '    nodesep=0.6;',
            '    ranksep=0.8;',
            '    node [shape=box, style="rounded,filled", fontname="SF Pro Display, Helvetica, Arial", fontsize=13, fontcolor="#ffffff", margin="0.25,0.15", width=1.5];',
            '    edge [fontname="Helvetica", penwidth=2.5, arrowsize=0.8];',
            '',
        ]
        
        # __start__ node (always active)
        dot_lines.append(f'    start [label="__start__", fillcolor="{active_color}", color="{active_color}"];')
        
        # agent node (always active)
        dot_lines.append(f'    agent [label="agent", fillcolor="{active_color}", color="{active_color}"];')
        
        # tools node (active only if used)
        tools_color = active_color if used_tools else inactive_color
        dot_lines.append(f'    tools [label="tools", fillcolor="{tools_color}", color="{tools_color}"];')
        
        # __end__ node (always active)
        dot_lines.append(f'    end [label="__end__", fillcolor="{active_color}", color="{active_color}"];')
        
        dot_lines.append('')
        
        # start -> agent (always active)
        dot_lines.append(f'    start -> agent [color="{edge_active}"];')
        
        # agent -> tools (active if used)
        agent_tools_color = edge_active if used_tools else edge_inactive
        dot_lines.append(f'    agent -> tools [color="{agent_tools_color}"];')
        
        # tools -> agent (active if used)
        dot_lines.append(f'    tools -> agent [color="{agent_tools_color}"];')
        
        # agent -> end (always active)
        dot_lines.append(f'    agent -> end [color="{edge_active}"];')
        
        dot_lines.append("}")
        return "\n".join(dot_lines)
    
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