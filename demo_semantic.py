import os
import time
from datetime import datetime
from dotenv import load_dotenv
_ = load_dotenv()

# Timing utility functions
def print_timestamp(label=""):
    """Print current timestamp with optional label"""
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
    print(f"[{current_time}] {label}")

def time_execution(func_name="Operation"):
    """Context manager or decorator for timing operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            print_timestamp(f"Starting {func_name}")
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            print_timestamp(f"Completed {func_name} - Duration: {duration:.3f} seconds")
            return result
        return wrapper
    return decorator

openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o-mini", temperature=0.0)

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.store.memory import InMemoryStore

from prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt

#####################Local Chat Ollama model######################
# Uncomment the following lines to use a local Ollama model instead of OpenAI
#llm = ChatOllama(model="llama3.2:latest", temperature=0.0)
######################Local Chat Ollama model######################
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}

email = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "body": """
Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )


llm_router = llm.with_structured_output(Router)

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    print_timestamp(f"Tool: write_email called for {to}")
    # Placeholder response - in real app would send email
    time.sleep(0.1)  # Simulate some processing time
    result = f"Email sent to {to} with subject '{subject}'"
    print_timestamp(f"Tool: write_email completed")
    return result

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    print_timestamp(f"Tool: schedule_meeting called for {preferred_day}")
    # Placeholder response - in real app would check calendar and schedule
    time.sleep(0.2)  # Simulate some processing time
    result = f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"
    print_timestamp(f"Tool: schedule_meeting completed")
    return result

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    print_timestamp(f"Tool: check_calendar_availability called for {day}")
    # Placeholder response - in real app would check actual calendar
    time.sleep(0.15)  # Simulate some processing time
    result = f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"
    print_timestamp(f"Tool: check_calendar_availability completed")
    return result


store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"},
)

from langmem import create_manage_memory_tool, create_search_memory_tool

manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant", 
        "{langgraph_user_id}",
        "collection"
    )
)
search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)


# print(manage_memory_tool.name)
# print()
# print(manage_memory_tool.description)
# print()
# print(manage_memory_tool.args)
# print()
# print(search_memory_tool.name)
# print()
# print(search_memory_tool.description)
# print()
# print(search_memory_tool.args)
# print()


agent_system_prompt_memory = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {name}'s communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""

def create_prompt(state):
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt_instructions["agent_instructions"], 
                **profile
            )
        }
    ] + state['messages']

from langgraph.prebuilt import create_react_agent

response_agent = create_react_agent(
    llm,
    tools=[
        write_email, 
        schedule_meeting, 
        check_calendar_availability, 
        manage_memory_tool, 
        search_memory_tool
    ],
    prompt=create_prompt,
    store=store,
)

config = {"configurable": {"langgraph_user_id": "lance"}}

# Timing the first agent invocation
print_timestamp("About to invoke agent with 'Jim is my friend'")
start_time = time.perf_counter()

response = response_agent.invoke(
    {"messages": [{"role": "user", "content": "Jim is my friend"}]},
    config=config # type: ignore
)

end_time = time.perf_counter()
duration = end_time - start_time
print_timestamp(f"First agent invocation completed - Duration: {duration:.3f} seconds")

# for m in response["messages"]:
#     m.pretty_print()

# Timing the second agent invocation
print_timestamp("About to invoke agent with 'who is Jim?'")
start_time2 = time.perf_counter()

response2 = response_agent.invoke(
    {"messages": [{"role":"user","content":"who is Jim?"}]},
    config=config # type: ignore
)

end_time2 = time.perf_counter()
duration2 = end_time2 - start_time2
print_timestamp(f"Second agent invocation completed - Duration: {duration2:.3f} seconds")

# print_timestamp("Printing response messages")
# for m in response2["messages"]:
#     m.pretty_print()

# print_timestamp("Checking store namespaces")
# print("--------------------------------------------------")
# print(store.list_namespaces())
# print_timestamp("Script execution completed")

# print("--------------------------------------------------")
# print(store.search(('email_assistant', 'lance', 'collection'), query='Jim'))