from langchain_google_community import GmailToolkit
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

toolkit = GmailToolkit()

tools = toolkit.get_tools()

llm = ChatOllama(model="llama3.2:latest", temperature=0.0)

agent_executor = create_react_agent(
    llm,
    tools,
    )

example_query = "Draft an email to fake@fake.com thanking them for coffee."

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

