
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
import base64


def main():

    # Function to insert file data into Snowflake
    def insert_file_data(cursor, file_id, filename, file_data):
        file_data_str = base64.b64encode(file_data).decode('utf-8')  # Convert binary data to string
        cursor.execute(f"""
            INSERT INTO {snowbotium_table_files} (id, filename, filedata)
            VALUES (%s, %s, %s)
        """, (file_id, filename, file_data_str))

    # Function to insert prompt-response data into Snowflake
    def insert_prompt_response(cursor, prompt_id, prompt, response):
        cursor.execute(f"""
            INSERT INTO {snowbotium_table_responses} (id, prompt, response)
            VALUES (%s, %s, %s)
        """, (prompt_id, prompt, response))

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


    # Function to try and load chat history items from JSON
    def load_chat_history_items() -> List[dict]:
        chat_history_items = []
        try:
            with open("chat_history.json", "r") as history_file:
                for line in history_file:
                    chat_history_items.append(json.loads(line.strip()))
        except FileNotFoundError:
            pass

        return chat_history_items


    # Snowflake connection parameters
    snowflake_user = st.secrets["snowflake"]["user"]
    snowflake_password = st.secrets["snowflake"]["password"]
    snowflake_account = st.secrets["snowflake"]["account"]
    snowflake_database = st.secrets["snowflake"]["database"]
    snowflake_schema = st.secrets["snowflake"]["schema"]
    snowbotium_table_files = "snowbotium_files"
    snowbotium_table_responses = "snowbotium_responses"

    # Initialize Snowflake connection
    try:
        with snowflake.connector.connect(
            user=snowflake_user,
            password=snowflake_password,
            account=snowflake_account,
            warehouse='COMPUTE_WH',
            database=snowflake_database,
            schema=snowflake_schema
        ) as conn:
            cursor = conn.cursor()

            # Create table to store file data
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {snowbotium_table_files} (
                    id VARCHAR(100),
                    filename VARCHAR(255),
                    filedata VARCHAR(16777216)
                )
            """)

            # Create table to store prompt-response data
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {snowbotium_table_responses} (
                    id INTEGER,
                    prompt VARCHAR(1000),
                    response VARCHAR(1000)
                )
            """)

            # Load chat history items
            chat_history_items = load_chat_history_items()

            # Select AI assistant and user roles
            assistant_role = select_role(role_type="Assistant", roles=["Bot", "Human"])
            user_role = select_role(role_type="User", roles=["Bot", "Human"])

            # Collect task name from user
            task_name = st.text_input("Enter a task name:")

            # Generate unique task name based on input and chat history
            unique_task_name = generate_unique_task_name(task_name, chat_history_items)

            # OpenAI models and prompt templates
            model_name = st.secrets["OPENAI_API_KEY"]["model_name"]
            api_key = st.secrets["OPENAI_API_KEY"]["api_key"]
            chat_model = ChatOpenAI(model_name=model_name, api_key=api_key)
            human_message_template = HumanMessagePromptTemplate.from_dict(st.secrets["prompts"]["human"])
            system_message_template = SystemMessagePromptTemplate.from_dict(st.secrets["prompts"]["system"])

            # Wrap entire Streamlit app within try-except block to catch any Streamlit errors
            try:

                # Streamlit app begins here
                st.title("CAMEL - Conversational AI with MLE")

                # Initialize CAMELAgent
                agent = CAMELAgent(
                    assistant_role=assistant_role,
                    user_role=user_role,
                    chat_model=chat_model,
                    human_message_template=human_message_template,
                    system_message_template=system_message_template,
                    chat_history_items=chat_history_items,
                    task_name=unique_task_name,
                    snowflake_cursor=cursor,
                    insert_file_data_func=insert_file_data,
                    insert_prompt_response_func=insert_prompt_response
                )


                # Streamlit app continues here
                st.write(f"Task Name: {unique_task_name}")
                st.write(f"Assistant Role: {assistant_role}")
                st.write(f"User Role: {user_role}")

                # Collect user input and generate AI response
                user_input = st.text_input("You >>>")
                ai_response = agent.generate_response(user_input)

                # Display AI response
                st.write(f"{user_role} >>> {ai_response}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    except snowflake.connector.errors.ProgrammingError as e:
        st.error(f"Error Connecting to Snowflake: {e.msg}")

if __name__ == "__main__":
    main()
