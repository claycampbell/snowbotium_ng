import streamlit as st
import os
import PyPDF2
import snowflake.connector
import pandas as pd
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
import uuid

# Snowflake connection parameters
snowflake_user = st.secrets["snowflake"]["user"]
snowflake_password = st.secrets["snowflake"]["password"]
snowflake_account = st.secrets["snowflake"]["account"]
snowflake_database = st.secrets["snowflake"]["database"]
snowflake_schema = st.secrets["snowflake"]["schema"]
snowbotium_table_files = "snowbotium_files"
snowbotium_table_responses = "snowbotium_responses"

# Initialize Snowflake connection
conn = snowflake.connector.connect(
    user=snowflake_user,
    password=snowflake_password,
    account=snowflake_account,
    warehouse='COMPUTE_WH',
    database=snowflake_database,
    schema=snowflake_schema
)
# Create Snowflake cursor
cursor = conn.cursor()

# Create Snowflake tables if they don't exist
cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {snowbotium_table_files} (
        id STRING,
        filename STRING,
        filedata VARCHAR
    )
""")

cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {snowbotium_table_responses} (
        id STRING,
        prompt STRING,
        response STRING
    )
""")

# Function to insert file data into Snowflake
def insert_file_data(file_id, filename, file_data):
    cursor.execute(f"""
        INSERT INTO {snowbotium_table_files} (id, filename, filedata)
        VALUES (%s, %s, %s)
    """, (file_id, filename, file_data))
    conn.commit()

# Function to insert prompt-response data into Snowflake
def insert_prompt_response(prompt_id, prompt, response):
    cursor.execute(f"""
        INSERT INTO {snowbotium_table_responses} (id, prompt, response)
        VALUES (%s, %s, %s)
    """, (prompt_id, prompt, response))
    conn.commit()

# Function to select roles
def select_role(role_type, roles):
    selected_role = st.selectbox(f"Select {role_type} role:",  ["Custom Role"] + roles )
    if selected_role == "Custom Role":
        custom_role = st.text_input(f"Enter the {role_type} (Custom Role):")
        return custom_role
    else:
        return selected_role

# Function to get system messages for AI assistant and AI user from role names and the task
def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]

    return assistant_sys_msg, user_sys_msg

def generate_unique_task_name(task: str, chat_history_items: List[dict]) -> str:
    task_id = str(uuid.uuid4())
    task_name = f"{task}_{task_id}"
    
    # Check if the task name already exists in the chat history items
    for item in chat_history_items:
        if item.get("task_name") == task_name:
            # If the task name already exists, generate a new unique task name
            return generate_unique_task_name(task, chat_history_items)
    
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

st.title("Automation Rodeo 🐂")

# Sidebar: API Key input
st.sidebar.title("Configuration")
# comment this out if you want to use the API key from the environment variable locally
#api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# uncomment this if you want to use the API key from the environment variable locally
api_key = os.getenv("OPENAI_API_KEY")

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
                    st.write(f"<p style='color: green; font-weight: bold;'>This App was used to solve *{tasks_solved}* tasks so far since deployed</p>", unsafe_allow_html=True)
# Main: Load roles from roles.txt
with open("roles.txt", "r") as roles_file:
    roles_list = [line.strip() for line in roles_file.readlines()]

# Main: Role selection
user_role_name = select_role("AI user", roles_list)
assistant_role_name = select_role("AI assistant", roles_list)

if assistant_role_name and user_role_name:
    # Main: Task input
    task = st.text_input("Please enter the task:")

    if task:
        # Main: Task specifier
        task_specifier = st.checkbox("Do you want to use the task specifier feature?", help="Use the task specifier feature to make a task more specific by GPT. May not work as expected.")

        if task_specifier:
            word_limit = st.number_input("Please enter the word limit for the specified task:", min_value=1, value=50, step=1)

            if word_limit:
                task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
                task_specifier_prompt = (
                    """Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
                    Please make it more specific. Be creative and imaginative.
                    Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
                )
                task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
                task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(model=model, temperature=1.0))
                task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                            user_role_name=user_role_name,
                                                            task=task, word_limit=word_limit)[0]
                specified_task_msg = task_specify_agent.step(task_specifier_msg)
                st.write(f"<p style='font-weight: bold;'>Specified task:</p> {specified_task_msg.content}", unsafe_allow_html=True)

                specified_task = specified_task_msg.content
        else:
            specified_task = task

        if specified_task:
            # Main: Chat turn limit input
            chat_turn_limit = st.number_input("Please enter the chat turn limit:", min_value=1, step=1)
            
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file is not None:
        # Read the uploaded PDF file
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                file_content = ""
                for page in pdf_reader.pages:
                    file_content += page.extract_text()
                insert_file_data(str(uploaded_file.name), uploaded_file.name, file_content)

            if st.button("Start Solving Task"):
                if api_key == "":
                    st.warning("Please enter your OpenAI API Key.")
                    st.stop()

                with open("stats.txt", "w") as stats_file:
                    stats_file.write(str(tasks_solved))

                chat_history_items = load_chat_history_items()
                with st.spinner("Thinking..."):
                    # Main: Initialize agents and start role-playing session
                    assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)
                    assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(model=model, temperature=0.2))
                    user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(model=model, temperature=0.2))

                    assistant_agent.reset()
                    user_agent.reset()

                    assistant_msg = HumanMessage(
                        content=(f"{user_sys_msg.content}. "
                                "Now start to give me introductions one by one. "
                                "Only reply with Instruction and Input."))

                    user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
                    user_msg = assistant_agent.step(user_msg)

                    st.write(f"<p style='color: red;'><b>Original task prompt:</b></p>\n\n{task}\n", unsafe_allow_html=True)
                    st.write(f"<p style='color: red;'><b>Specified task prompt:</b></p>\n\n{specified_task}\n", unsafe_allow_html=True)

                # Use the chat history, including file content, in the conversation
               # Use the chat history, including file content, in the conversation
                chat_history = []
                for idx, item in enumerate(chat_history_items):
                    if 'role' in item:
                        role = item['role']
                        content = item['content']
                        if role == assistant_role_name:
                            user_msg = HumanMessage(content=content)
                            user_msg = assistant_agent.step(user_msg)
                        else:
                            user_msg = HumanMessage(content=content)
                            user_msg = user_agent.step(user_msg)
                        chat_history.append({"role": assistant_role_name, "content": user_msg.content})

                        # Check if the maximum turn limit has been reached
                        if idx >= chat_turn_limit:
                            break
                    else:
                        # Handle the case where the 'role' key is missing
                        # You can raise an error, log a message, or take appropriate action
                        # Here, I'm printing a message for reference
                        print(f"Skipping item {item} because 'role' key is missing")

                # Append file_content to the chat_history
                chat_history.append({"role": user_role_name, "content": file_content})

                # Use the file_content as context in the conversation
                user_msg = HumanMessage(content=file_content)
                user_msg = assistant_agent.step(user_msg)
                chat_history.append({"role": assistant_role_name, "content": user_msg.content})

                # Print the modified chat history
                for item in chat_history:
                    print(f"{item['role']}: {item['content']}")

                # Retrieve the responses from the chat history
                responses = [item['content'] if 'content' in item else '' for item in chat_history[-len(file_content):]]

                with st.spinner("Running role-playing session to solve the task..."):
                    # Replace the for loop with the following code:
                    progress = st.progress(0)

                    for n in range(chat_turn_limit):
                        user_ai_msg = user_agent.step(assistant_msg)
                        user_msg = HumanMessage(content=user_ai_msg.content)

                        chat_history.append({"role": user_role_name, "content": user_msg.content, "file_content": file_content})
                        st.markdown(f"<p style='color: blue; font-weight: bold;'>{user_role_name}</p>\n\n{user_msg.content}\n\n", unsafe_allow_html=True)

                        assistant_ai_msg = assistant_agent.step(user_msg)
                        assistant_msg = HumanMessage(content=assistant_ai_msg.content)

                        chat_history.append({"role": assistant_role_name, "content": assistant_msg.content,"file_content": file_content})
                        st.markdown(f"<p style='color: green; font-weight: bold;'>{assistant_role_name}</p>\n\n{assistant_msg.content}\n\n", unsafe_allow_html=True)

                        progress.progress((n+1)/chat_turn_limit)

                        if "<CAMEL_TASK_DONE>" in user_msg.content:
                            break

                    progress.empty()

                # Assuming you have already established a connection to Snowflake
                conn = snowflake.connector.connect(
                    user='claycampbell',
                    password='Camps3116',
                    account='mi14164.ca-central-1.aws',
                    warehouse='COMPUTE_WH',
                    database='SNOWBOTIUM',
                    schema='public',
                    snowbotium_table_responses="automation_rodeo_tasks"
                )

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

                # Get the current conversation
                current_conversation = chat_history[-1]

                # Convert the current conversation to JSON format
                current_conversation_json = json.dumps(current_conversation)

                # Replace single quotes with double quotes in the conversation JSON
                current_conversation_json = current_conversation_json.replace("'", "''")

                # Create a DataFrame with the current conversation
                df = pd.DataFrame([current_conversation_json], columns=['conversation'])

                # Load DataFrame data into Snowflake table
                cursor = conn.cursor()

                # Create the table in Snowflake (if it doesn't exist)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS automation_rodeo_tasks (
                        id INT,
                        "task" VARIANT,
                        "assistant_role_name" VARCHAR(255),
                        "user_role_name" VARCHAR(255),
                        "model" VARCHAR(255),
                        "chat_turn_limit" INT,
                        "conversation" VARIANT
                    )
                """)

                # Insert DataFrame data into Snowflake table
                cursor.execute("""
                    INSERT INTO automation_rodeo_tasks (CONVERSATION)
                    SELECT PARSE_JSON(%s)
                """, (df.to_json(orient='records'),))

                # Commit the changes
                conn.commit()

                # Close the cursor
                cursor.close()

                # Close the connection
                conn.close()



            else:
                st.warning("Please enter the chat turn limit.")
        else:
            st.warning("Please specify the task.")
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
