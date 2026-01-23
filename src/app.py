import streamlit as st
import re
import time
from chatbot import BootcampChatbot  # Import from backend

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Bootcamp Tutor Chatbot",
    layout="wide"
)

# ------------------- HELPER FUNCTIONS -------------------
def extract_video_urls(text):
    """Extract YouTube and video URLs from text"""
    # YouTube patterns
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
    ]
    
    # Video file patterns
    video_patterns = [
        r'https?://[^\s]+\.(?:mp4|mov|avi|mkv|webm|flv)',
    ]
    
    urls = []
    
    # Check for YouTube URLs
    for pattern in youtube_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            video_id = match.group(1)
            urls.append(f"https://www.youtube.com/watch?v={video_id}")
    
    # Check for direct video URLs
    for pattern in video_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            urls.append(match.group(0))
    
    return urls

# ------------------- TITLE -------------------
st.title("Brown Bag: Powered-up! Chatbot")
st.caption("Powered by LangGraph & LangChain - Ask me to find videos!")

# ------------------- INITIALIZE SESSION STATE -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = BootcampChatbot()

# ------------------- SIDEBAR: AGENT GRAPH -------------------
with st.sidebar:
    st.markdown("### LangGraph Agent")
    st.caption("Real-time execution trace")
    sidebar_graph = st.empty()

def render_graph(current_node="idle", used_tools=False, trace=None):
    """Render the graph in sidebar"""
    with sidebar_graph.container():
        if current_node == "idle" and trace:
            # Show completed graph from trace (using live graph at "end" state)
            trace_used_tools = any(s["node"] in ["Agent Decision", "Tool Execution"] for s in trace)
            dot_graph = st.session_state.chatbot.get_live_graph_dot("end", trace_used_tools)
            if dot_graph:
                st.graphviz_chart(dot_graph)
        elif current_node != "idle":
            # Show live animated graph
            dot_graph = st.session_state.chatbot.get_live_graph_dot(current_node, used_tools)
            if dot_graph:
                st.graphviz_chart(dot_graph)
        else:
            st.info("Ask a question to see the agent in action!")

# Initial render
latest_trace = None
for msg in reversed(st.session_state.messages):
    if msg.get("role") == "assistant" and msg.get("trace"):
        latest_trace = msg["trace"]
        break
render_graph(trace=latest_trace)


# ------------------- DISPLAY CHAT HISTORY -------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If message contains videos, display them
        if "videos" in message:
            for video_url in message["videos"]:
                st.video(video_url)

# ------------------- CHAT INPUT -------------------
if prompt := st.chat_input("Ask me anything... or ask me to find a video!"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("*Processing...*")
        
        try:
            # LIVE animated execution
            bot_message = None
            trace_steps = []
            used_tools = False
            
            for state in st.session_state.chatbot.chat_with_live_trace(prompt, "streamlit_session"):
                current_node = state[0]
                used_tools = state[1]
                is_complete = state[2]
                
                # Update sidebar graph LIVE
                render_graph(current_node, used_tools)
                time.sleep(0.4)  # Pause so audience can see each step
                
                if is_complete:
                    bot_message = state[3]
                    trace_steps = state[4] if len(state) > 4 else []
            
            # Show final response
            response_placeholder.markdown(bot_message)
            
            # Show final graph state
            render_graph("end", used_tools)
            
            # Extract and display videos from response
            video_urls = extract_video_urls(bot_message)
            
            if video_urls:
                st.success(f"Found {len(video_urls)} video(s)")
                for url in video_urls:
                    st.video(url)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": bot_message,
                    "videos": video_urls,
                    "trace": trace_steps
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": bot_message,
                    "trace": trace_steps
                })
        
        except Exception as e:
            response_placeholder.empty()
            st.error(f"Error: {str(e)}")
            st.info("Make sure your api key is set in your .env file!")


st.divider()
st.caption("Built with using Streamlit, LangChain & LangGraph")