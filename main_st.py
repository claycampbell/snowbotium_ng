import streamlit as st
import os
import PyPDF2
import snowflake.connector
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

import base64

# Function to insert file data into Snowflake
def insert_file_data(file_id, filename, file_data):
    file_data_str = base64.b64encode(file_data).decode('utf-8')  # Convert binary data to string
    cursor.execute(f"""
        INSERT INTO {snowbotium_table_files} (id, filename, filedata)
        VALUES (%s, %s, %s)
    """, (file_id, filename, file_data_str))
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
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name)

# Function to chat with agents
def chat_with_agents(agent_1: CAMELAgent, agent_2: CAMELAgent):
    st.write("Welcome to the agent chat!")
    st.write("You can start the conversation by sending a message as the user.")
    st.write("Type '/end' to end the conversation.")

    user_message = st.text_input("User:")
    while user_message.strip() != "/end":
        if user_message.strip() != "":
            response_1 = agent_1.send_message(user_message)
            agent_1_message = response_1["message"]
            st.write(f"Agent 1: {agent_1_message}")

            response_2 = agent_2.send_message(agent_1_message)
            agent_2_message = response_2["message"]
            st.write(f"Agent 2: {agent_2_message}")

        user_message = st.text_input("User:")

# Main function
def main():
    st.title("AI Agent Communication")

    st.sidebar.title("Settings")

    task = st.sidebar.text_input("Task")
    assistant_name = st.sidebar.text_input("Assistant Name")
    user_name = st.sidebar.text_input("User Name")
    roles = [
        "assistant",
        "user",
        "system"
    ]

    assistant_role_name = select_role("Assistant", roles)
    user_role_name = select_role("User", roles)

    if task:
        sys_msgs = get_sys_msgs(assistant_role_name, user_role_name, task)
    else:
        st.sidebar.error("Please enter a value for the 'Task' field.")

    if st.sidebar.button("Start Chat"):   
        agent_1 = CAMELAgent()
        agent_2 = CAMELAgent()

        # Set the agent roles and system messages
        agent_1.set_role(assistant_role_name)
        agent_2.set_role(user_role_name)
        agent_1.set_sys_msgs(sys_msgs)
        agent_2.set_sys_msgs(sys_msgs)

        # Chat with the agents
        chat_with_agents(agent_1, agent_2)

if __name__ == "__main__":
    main()
