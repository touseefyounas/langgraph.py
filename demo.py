from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from langgraph.graph import add_messages

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal

profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_description": "A software developer with a passion for AI and machine learning.",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently. Always provide a final response after using tools."
}

# Example incoming email
email = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """
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


llm = ChatOllama(model="llama3.2:latest", temperature=0.0)

llm_router = llm.with_structured_output(Router)

system_prompt = triage_system_prompt.format(
    full_name=profile["full_name"],
    name=profile["name"],
    examples=None,
    user_profile_background=profile["user_profile_description"],
    triage_no=prompt_instructions["triage_rules"]["ignore"],
    triage_notify=prompt_instructions["triage_rules"]["notify"],
    triage_email=prompt_instructions["triage_rules"]["respond"],
    )

user_prompt = triage_user_prompt.format(
    author=email["author"],
    to=email["to"],
    subject=email["subject"],
    email_thread=email["email_thread"],
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

# print(llm_router.invoke(messages))

# for chunk in llm_router.stream(messages):
#     print(str(chunk), end="")

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """write and send an email"""
    #placeholder response simulating email sending
    return f"Email sent to {to} with subject '{subject}' and content: {content}" 


@tool
def schedule_meeting(
    attendees: list[str],
    subject: str,
    duration_minutes: int,
    preferred_day: str,
) -> str:
    """Schedule a meeting with the given attendees."""
    # Placeholder response simulating meeting scheduling
    return f"Meeting scheduled with {', '.join(attendees)} on {preferred_day} for {duration_minutes} minutes about '{subject}'."

@tool
def check_calendar_availability(day: str) -> str:
    """Check available time slots for a given day."""
    # Placeholder response simulating calendar availability check
    return f"Available time slots for {day} are 9 AM - 11 AM, 1 PM - 3 PM."

def create_prompt(state):
    return [
        {
            "role": "system", 
            "content": agent_system_prompt.format(
                instructions=prompt_instructions["agent_instructions"],
                **profile
                )
        }
    ] + state['messages']


tools = [write_email, schedule_meeting, check_calendar_availability]


agent = create_react_agent(
    llm,
    tools=tools,
    prompt=create_prompt,
)

# response = agent.invoke(
#     {"messages":[{
#         "role": "user", 
#         "content": "What's my availability for tuesday?"
#     }]}
# )

# print(response["messages"][-1])

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    author = state["email_input"]["author"]
    to = state["email_input"]["to"]
    subject = state["email_input"]["subject"]
    email_thread = state["email_input"]["email_thread"]

    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        examples=None,
        user_profile_background=profile["user_profile_description"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
    )

    user_prompt = triage_user_prompt.format(
        author=author,
        to=to,
        subject=subject,
        email_thread=email_thread,
    )

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    
    if result.classification  == "respond": #type: ignore
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification  == "ignore": #type: ignore
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification  == "notify": #type: ignore
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification }") #type: ignore
    return Command(goto=goto, update=update) # type: ignore

# Create the state graph for the email agent

def build_email_agent():
    email_agent = StateGraph(State)
    email_agent = email_agent.add_node(triage_router)
    email_agent = email_agent.add_node("response_agent", agent)
    email_agent = email_agent.add_edge(START, "triage_router")
    return email_agent.compile()


from dummy_emails import email_input1, email_input2

# response = build_email_agent().invoke(
#     {"email_input": email_input2,}
# )
# print(response)

if __name__ == "__main__":
    print("Starting email triage...")
    response = build_email_agent().invoke(
        {"email_input": email_input1,}
    )
    for m in response["messages"]:
        m.pretty_print()
























# print("Starting Ollama chat...")
# while True:
#     try:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Exiting chat...")
#             break
#         response = llm.invoke(user_input)
#         print(f"Ollama: {response.content}")
#     except KeyboardInterrupt:
#         print("\nExiting chat...")
#         break
#     except Exception as e:
#         print(f"An error occurred: {e}")