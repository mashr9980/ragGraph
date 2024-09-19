from System_Robustness_and_Performance import LoadBalancer
import os
from dotenv import load_dotenv

load_dotenv()

neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

lb = LoadBalancer(neo4j_uri, neo4j_user, neo4j_password)
lb.initialize()
print(f"Query engines: {lb.query_engines}")