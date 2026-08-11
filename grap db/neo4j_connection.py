import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. SETUP AND AUTHENTICATION
# ==========================================
# Load environment variables from a .env file (if you have one)
load_dotenv()

# Replace these with your actual credentials if not using a .env file
os.environ["GROQ_API_KEY"] = "gsk_your_groq_api_key"
NEO4J_URI = "bolt://localhost:7687" # or neo4j+s://... for Aura
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_secure_password"

print("Connecting to Neo4j...")
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)
print("✅ Connected to Neo4j successfully!")

# ==========================================
# 2. INSERT DATA INTO NEO4J (The "Mapping" Phase)
# ==========================================
# Here we write the Cypher query that builds the database.
# We will use the classic Movies and Actors dataset.
seed_query = """
MERGE (m1:Movie {title: 'The Matrix'})
SET m1.released = 1999, m1.genre = 'Sci-Fi'

MERGE (m2:Movie {title: 'John Wick'})
SET m2.released = 2014, m2.genre = 'Action'

MERGE (p1:Person {name: 'Keanu Reeves'})
SET p1.born = 1964

MERGE (p2:Person {name: 'Carrie-Anne Moss'})
SET p2.born = 1967

MERGE (p3:Person {name: 'Chad Stahelski'})

// Creating the relationships
MERGE (p1)-[:ACTED_IN {role: 'Neo'}]->(m1)
MERGE (p2)-[:ACTED_IN {role: 'Trinity'}]->(m1)
MERGE (p1)-[:ACTED_IN {role: 'John Wick'}]->(m2)
MERGE (p3)-[:DIRECTED]->(m2)
"""

print("\nSeeding the database with Cypher...")
graph.query(seed_query)
# Refresh the schema so LangChain reads the newly inserted data format
graph.refresh_schema()
print("✅ Database populated and schema refreshed!")

# ==========================================
# 3. BUILD THE LLM GRAPH QA CHAIN
# ==========================================
print("\nInitializing Groq LLM and GraphCypherQAChain...")
# Initialize the Groq LLM. We use temperature=0 for precise Cypher generation.
llm = ChatGroq(
    model="llama3-8b-8192", # You can also use "mixtral-8x7b-32768" or other Groq models
    temperature=0
)

# Build the chain that bridges the LLM and the Neo4j database
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True, # This lets us see the generated Cypher query in the console
    allow_dangerous_requests=True # Required by LangChain to allow database querying
)
print("✅ QA Application is ready!")

# ==========================================
# 4. QUERY THE DATABASE WITH NATURAL LANGUAGE
# ==========================================
print("\n--- Asking Questions ---")

# Question 1
question_1 = "Who acted in The Matrix and what roles did they play?"
print(f"Question: {question_1}")
response_1 = chain.invoke({"query": question_1})
print(f"\nFinal LLM Explanation:\n{response_1['result']}\n")

print("-" * 40)

# Question 2
question_2 = "Did Keanu Reeves act in any movies directed by Chad Stahelski?"
print(f"Question: {question_2}")
response_2 = chain.invoke({"query": question_2})
print(f"\nFinal LLM Explanation:\n{response_2['result']}")

# ==========================================
# 5. GENERATE CYPHER TO INSERT DATA USING LANGCHAIN
# ==========================================
print("\n--- Generating Cypher to Insert New Data ---")
insert_prompt = ChatPromptTemplate.from_template(
    "You are an expert Neo4j Cypher developer. Given the following natural language description of data, "
    "generate a single valid Neo4j Cypher query to insert this data into the database using MERGE statements. "
    "Use 'Movie' and 'Person' node labels. "
    "Do not include any explanations, markdown formatting, or introductory text. Return ONLY the raw Cypher query.\n\n"
    "Description: {description}"
)

# Create a chain to generate the Cypher query
cypher_generator = insert_prompt | llm | StrOutputParser()

# Natural language description of the data to insert
data_description = (
    "A new movie called 'Inception' was released in 2010. It is a Sci-Fi movie. "
    "Leonardo DiCaprio acted in it as 'Cobb'. Christopher Nolan directed it."
)
print(f"Data Description: {data_description}")

# 1. Generate the query with Langchain
generated_cypher = cypher_generator.invoke({"description": data_description})

# Clean up any potential markdown formatting the LLM might have added
generated_cypher = generated_cypher.replace("```cypher", "").replace("```", "").strip()
print(f"\nGenerated Cypher Query:\n{generated_cypher}\n")

# 2. Insert it into Neo4j
print("Executing generated query to insert data...")
graph.query(generated_cypher)
graph.refresh_schema()
print("✅ New data inserted and schema refreshed!")

# 3. Verify it works with our QA chain
question_3 = "Who directed Inception and who acted in it?"
print(f"\nQuestion: {question_3}")
response_3 = chain.invoke({"query": question_3})
print(f"\nFinal LLM Explanation:\n{response_3['result']}")