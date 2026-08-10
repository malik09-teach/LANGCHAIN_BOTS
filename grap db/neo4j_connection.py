import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# Force reload the environment variables from the .env file
load_dotenv(override=True)

password = os.getenv("NEO4J_PASSWORD")
user_name = os.getenv("NEO4J_USERNAME")
uri = os.getenv("NEO4J_URI")

try:
    print(f"Attempting to connect to Neo4j at: {uri} as {user_name}...")
    
    # Initialize the Neo4j connection using the correct and updated LangChain class
    graph = Neo4jGraph(
        url=uri,
        username=user_name,
        password=password
    )
    
    # Refresh the graph schema to verify the connection works
    graph.refresh_schema()
    
    print("✅ Successfully connected to Neo4j database!")
    print("\nDatabase Schema:")
    print(graph.schema)
    
except Exception as e:
    print("❌ Failed to connect to Neo4j.")
    print(f"Error details: {e}")
