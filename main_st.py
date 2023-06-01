import streamlit as st
import os
from termcolor import colored
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from camel_agent import CAMELAgent
from inception_prompts import assistant_inception_prompt, user_inception_prompt
import json


# Function to select roles
def select_role(role_type, roles):
    selected_role = st.selectbox(f"Select {role_type} role:", ["Custom Role"] + roles)
    if selected_role == "Custom Role":
        custom_role = st.text_input(f"Enter the {role_type} (Custom Role):")
        return custom_role
    else:
        return selected_role


# Function to get system messages for AI assistant and AI user from role names and the task
def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name,
                                                               user_role_name=user_role_name, task=task)[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(assistant_role_name=assistant_role_name,
                                                     user_role_name=user_role_name, task=task)[0]

    return assistant_sys_msg, user_sys_msg


def generate_unique_task_name(task: str, chat_history_items: List[dict]) -> str:
    task_name = task
    count = 1
    task_names = [item["task"] for item in chat_history_items]

    while task_name in task_names:
        task_name = f"{task} ({count})"
        count += 1

    return task_name


def load_chat_history_items() -> List[dict]:
    chat_history_items = []
    try:
        with open("chat_history.json", "r") as history_file:
            for line in history_file:
                chat_history_items.append(json.loads(line.strip()))
    except FileNotFoundError:
        pass

    return chat_history_items


chat_history_items = []


st.set_page_config(layout="centered")

st.title("OmniSolver 🔆",
         help="This app uses the CAMEL framework to solve problems. This app uses GPT models and the responses may not be accurate")

# Sidebar: API Key input
st.sidebar.title("Configuration")
# comment this out if you want to use the API key from the environment variable locally
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# uncomment this if you want to use the API key from the environment variable locally
# api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
elif api_key == "":
    st.sidebar.warning("Please enter your OpenAI API Key.")


# Sidebar: Model selection
model = st.sidebar.radio("Select the model:", ("gpt-3.5-turbo", "gpt-4"))

with open("stats.txt", "r") as stats_file:
    stats = stats_file.readlines()
    tasks_solved = int(stats[0].strip())
    tasks_solved += 1
    st.write(f"<p style='color: green; font-weight: bold;'>This App was used to solve *{tasks_solved}* tasks so far since deployed</p>",
             unsafe_allow_html=True)
    stats[0] = str(tasks_solved) + "\n"
    with open("stats.txt", "w") as stats_file_write:
        stats_file_write.writelines(stats)

if model == "gpt-4":
    st.warning("The GPT-4 model is currently not available. Please select GPT-3.5 Turbo.")
    st.stop()

# Load the chat history items
chat_history_items = load_chat_history_items()

# Chat history list
chat_history = []

# Get the assistant and user role names
assistant_role_name = select_role("Assistant", ["Assistant"])
user_role_name = select_role("User", ["User"])

# Get the task name from the user
task = st.text_input("Enter the task you want to solve:", key="task_input")

if task:
    task_name = generate_unique_task_name(task, chat_history_items)
    st.info(f"Task: {task_name}")

    # Get system messages for AI assistant and AI user
    assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, task_name)

    # Create the AI assistant agent
    assistant_agent = CAMELAgent()

    # Start the conversation
    if st.button("Start"):
        chat_history.append(assistant_sys_msg)
        chat_history.append(user_sys_msg)

    # Conversation loop
    while True:
        # Display the chat history
        for item in chat_history:
            role = item["role"]
            content = item["content"]
            if role == assistant_role_name:
                st.text_area(role, content, key=f"{role}_history", height=200)
            else:
                st.text_area(role, content, key=f"{role}_history", height=100)

        # User input
        user_input = st.text_input(f"{user_role_name}:",
                                   help="Enter your message here and press Enter to send.", key="user_input")

        if st.button("Send") or user_input.endswith("\n"):
            # Add user message to chat history
            user_msg = HumanMessage(content=user_input.strip())
            chat_history.append(user_msg)

            # Reset conversation agents
            assistant_agent.reset()

            # Send user message to assistant agent
            assistant_msg = assistant_agent.step(user_msg)
            chat_history.append(assistant_msg)

            # Clear user input
            user_input = ""

        # Check if there is an uploaded file
        uploaded_file = st.file_uploader("Upload a document")

        # Handle the uploaded document
        if uploaded_file is not None:
            document_contents = uploaded_file.read().decode("utf-8")
        else:
            document_contents = ""

        # Include document in the conversation
        if document_contents:
            document_sys_msg = SystemMessage(content="Here is the document we will be working from:")
            assistant_msg = assistant_agent.step(document_sys_msg)
            user_msg = HumanMessage(content=document_contents)
            user_msg = assistant_agent.step(user_msg)
            chat_history.append({"role": "Document", "content": document_contents})

        # Continue with the existing code for the conversation loop
        if user_input.strip() == "":
            continue

        # AI User message
        user_msg = HumanMessage(content=user_input)
        chat_history.append(user_msg)

        # Reset conversation agents
        assistant_agent.reset()

        # AI User agent step
        assistant_msg = assistant_agent.step(user_msg)
        chat_history.append(assistant_msg)

else:
    st.info("Enter a task to start the conversation.")

# Main: Save chat history to file
task_name = generate_unique_task_name(task, chat_history_items)
history_dict = {
    "task": task_name,
    "settings": {
        "assistant_role_name": assistant_role_name,
        "user_role_name": user_role_name,
        "model": model,
        "chat_turn_limit": chat_turn_limit,
    },
    "conversation": chat_history,
}

with open("chat_history.json", "a") as history_file:
    json.dump(history_dict, history_file)
    history_file.write("\n")

else:
    st.warning("Please enter the task.")
else:
    st.warning("Please select both AI assistant and AI user roles.")

# Sidebar: Load chat history
chat_history_titles = [item["task"] for item in chat_history_items]
try:
    chat_history_items = load_chat_history_items()
    chat_history_titles = [item["task"] for item in chat_history_items]
    selected_history = st.sidebar.selectbox("Select chat history:", ["None"] + chat_history_titles)

    if selected_history != "None":
        delete_history_button = st.sidebar.button("Delete Selected Chat History")

        if delete_history_button and selected_history != "None":
            chat_history_items.pop(chat_history_titles.index(selected_history))

            # Save the updated chat history to file
            with open("chat_history.json", "w") as history_file:
                for item in chat_history_items:
                    json.dump(item, history_file)
                    history_file.write("\n")

            st.sidebar.success("Selected chat history deleted.")
            st.experimental_rerun()

    # Main: Display selected chat history
    if selected_history != "None":
        selected_history_item = chat_history_items[chat_history_titles.index(selected_history)]
        settings = selected_history_item["settings"]
        conversation = selected_history_item["conversation"]

        st.write(f"<p style='color: green; font-weight: bold;'>Task:</p> {selected_history}\n", unsafe_allow_html=True)

        st.write(f"""<p style='color: green; font-weight: bold;'>Settings:</p>
                    <p>- AI assistant role: <span >{settings['assistant_role_name']}</span></p>
                    <p>- AI user role: <span >{settings['user_role_name']}</span></p>
                    <p>- Model: {settings['model']}</p>
                    <p>- Chat turn limit: {settings['chat_turn_limit']}</p>
                    """, unsafe_allow_html=True)

        for msg in conversation:
            st.markdown(f"<p style='color: green; font-weight: bold;'>{msg['role']}</p>\n\n{msg['content']}\n\n", unsafe_allow_html=True)

except FileNotFoundError:
    st.sidebar.warning("No chat history available.")

