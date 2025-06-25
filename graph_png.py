from demo import build_email_agent

def display_email_agent_graph():
    email_agent = build_email_agent()
    
    # Generate the graph in PNG format
    graph_png = email_agent.get_graph(xray=True).draw_mermaid_png()
    
    # Save the graph to a file
    with open("email_agent_graph.png", "wb") as f:
        f.write(graph_png)
    
    print("Email agent graph saved as 'email_agent_graph.png'.")


display_email_agent_graph()