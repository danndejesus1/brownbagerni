import streamlit as st
import re
from chatbot import BootcampChatbot  # Import from backend

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Bootcamp Tutor Chatbot",
    page_icon="🤖",
    layout="centered"
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
st.title("🤖 Brown Bag: Chatbots")
st.caption("Powered by LangGraph & LangChain - Ask me to find videos!")

# ------------------- INITIALIZE SESSION STATE -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = BootcampChatbot()

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chatbot.clear_history("streamlit_session")
        st.rerun()

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
    
    # Get bot response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call backend chatbot
                bot_message = st.session_state.chatbot.chat(prompt, "streamlit_session")
                st.markdown(bot_message)
                
                # Extract and display videos from response
                video_urls = extract_video_urls(bot_message)
                
                if video_urls:
                    st.success(f"🎬 Found {len(video_urls)} video(s)!")
                    for url in video_urls:
                        st.video(url)
                    
                    # Add assistant message with videos to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": bot_message,
                        "videos": video_urls
                    })
                else:
                    # Add assistant message without videos
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": bot_message
                    })
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure your api key is set in your .env file!")

# ------------------- FOOTER -------------------
st.divider()
st.caption("Built with ❤️ using Streamlit, LangChain & LangGraph")