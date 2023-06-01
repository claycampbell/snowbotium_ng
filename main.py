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

import os
from camel_agent import CAMELAgent
from termcolor import colored

# define your apit key as an environment variable
# os.environ["OPENAI_API_KEY"] = ""



from termcolor import colored


def select_role(role_type, roles):
    print(colored(f"Available {role_type} Roles:", "yellow"))
    [print(colored(f"{i + 1}. {role}", "cyan")) for i, role in enumerate(roles)]
    print(colored(f"{len(roles) + 1}. Custom Role", "cyan"))
    
    choice = int(input(colored(f"Choose the {role_type} (or select Custom Role): ", "yellow")))
    return roles[choice - 1] if choice <= len(roles) else input(colored(f"Enter the {role_type} (Custom Role): ", "yellow"))

roles_list = ["Accountant", "Designer", "Python Programmer", "Teacher", "Web Developer"]

assistant_role_name = select_role("AI assistant", roles_list)
user_role_name = select_role("AI user", roles_list)

task = input(colored("Please enter the task: ", "yellow")) # e.g. Develop a trading bot for the stock market



# ask user if they want to use the task specifier feature
task_specifier = input("Do you want to use the task specifier feature? (y/n): ").lower()
if task_specifier == "y":
    word_limit = input(colored("Please enter the word limit for the specified task: ", "yellow")) # e.g. 50  # word limit for task brainstorming

    # task specifier is used to make a task more specific
    task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
    task_specifier_prompt = (
    """Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
    Please make it more specific. Be creative and imaginative.
    Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
    )
    task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
    task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=1.0))
    task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                                user_role_name=user_role_name,
                                                                task=task, word_limit=word_limit)[0]
    specified_task_msg = task_specify_agent.step(task_specifier_msg)
    print(f"Specified task: {specified_task_msg.content}")
    specified_task = specified_task_msg.content
else:
    specified_task = task


# import the prompts for assistant and user from inception_prompts.py
from inception_prompts import assistant_inception_prompt, user_inception_prompt

# get system messages for AI assistant and AI user from role names and the task
def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
    
    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
    
    return assistant_sys_msg, user_sys_msg


# Create AI assistant agent and AI user agent from obtained system messages
assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)
assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats 
assistant_msg = HumanMessage(
    content=(f"{user_sys_msg.content}. "
                "Now start to give me introductions one by one. "
                "Only reply with Instruction and Input."))

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
user_msg = assistant_agent.step(user_msg)


# Start role-playing session to solve the task!
print(colored(f"Original task prompt:\n{task}\n", "blue"))
print(colored(f"Specified task prompt:\n{specified_task}\n", "green"))


chat_turn_limit, n = int(input("Please enter the chat turn limit: ")), 0
while n < chat_turn_limit:
    n += 1
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    print(colored(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n", "magenta"))

    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    print(colored(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n", "cyan"))
    if "<CAMEL_TASK_DONE>" in user_msg.content:
        break
