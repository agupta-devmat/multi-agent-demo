import streamlit as st
import logging
from strands import Agent
from strands.models import BedrockModel
from strands_tools.a2a_client import A2AClientToolProvider
from dotenv import load_dotenv
from io import StringIO
import sys

import nest_asyncio
nest_asyncio.apply()

# --- 1. Initial Setup and Configuration ---

# Load environment variables from .env file for AWS credentials
load_dotenv(override=True)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the URL for the supervisor agent
SUPERVISOR_URL = "http://localhost:5000"

# Set the title and icon for the Streamlit web page
st.set_page_config(page_title="Multi-Agent System UI", page_icon="🤖")

st.title("🤖 Multi-Agent System Interface")
st.caption("Interact with a supervisor agent that orchestrates specialized tasks.")


# --- 2. Agent Initialization with Caching ---

# Use st.cache_resource to initialize the agent only once.
# This prevents re-connecting to the supervisor on every user interaction,
# making the app much faster and more efficient.
@st.cache_resource
def initialize_client_agent():
    """
    Discovers the supervisor agent's tools and initializes a local client agent
    that can use them.
    """
    logger.info(f"Attempting to discover agent at {SUPERVISOR_URL}...")
    try:
        # A2AClientToolProvider connects to the supervisor and discovers its capabilities
        provider = A2AClientToolProvider(known_agent_urls=[SUPERVISOR_URL])
    except Exception as e:
        # If connection fails, display an error in the UI and stop.
        st.error(f"Failed to connect to Supervisor Agent at {SUPERVISOR_URL}.")
        st.error("Please ensure supervisor_agent.py is running.")
        st.error(f"Details: {e}")
        return None

    if not provider.tools:
        st.error(f"No tools discovered from the agent at {SUPERVISOR_URL}.")
        st.error("Ensure the Supervisor Agent has defined tools.")
        return None

    logger.info("Successfully discovered the following capabilities:")
    for tool in provider.tools:
        logger.info(f"- {tool.__name__}")

    # Create the local "client" agent that will process user requests
    client_bedrock_model = BedrockModel(
        model_id="eu.anthropic.claude-sonnet-4-20250514-v1:0"
    )

    client_agent = Agent(
        system_prompt="""You are a command-line assistant. Your job is to
        analyze the user's instruction and execute the most appropriate tool.
        Use the tools available to you to find the supervisor agent and delegate the task to it.""",
        tools=provider.tools,
        model=client_bedrock_model
    )

    st.success("Successfully connected to the Supervisor Agent.")
    return client_agent


# --- 3. Main Application Logic ---

# Initialize the agent. This will be cached after the first run.
client_agent = initialize_client_agent()

# Initialize chat history in Streamlit's session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages from the history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The main application logic runs only if the agent was initialized successfully
if client_agent:
    # Use st.chat_input to get user input at the bottom of the screen
    if prompt := st.chat_input("Ask the agent to perform a multi-step task..."):
        # Add user's message to the chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display the assistant's response
        with st.chat_message("assistant"):
            # Use st.status to show the agent's thought process while it's working
            with st.status("Thinking and executing task...", expanded=True) as status:
                # Redirect stdout to capture the agent's detailed logs
                old_stdout = sys.stdout
                redirected_output = StringIO()
                sys.stdout = redirected_output

                # --- Execute the agent ---
                # This is the call that sends the request to your agentic framework
                final_response = client_agent(prompt)

                # Restore the original stdout
                sys.stdout = old_stdout

                # Get the captured thoughts and logs from the agent's execution
                agent_thoughts = redirected_output.getvalue()

                # Display the captured logs inside the status box
                status.markdown("#### Agent's Thought Process:")
                status.code(agent_thoughts, language="log")
                status.update(label="Task Complete!", state="complete", expanded=False)

            # Display the final answer from the agent
            st.markdown(final_response)
            # Add the assistant's final response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": final_response})
else:
    st.warning("Agent system is not available. Please resolve the errors above.")