from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import json

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(user, password))

def add_complex_data(tx, query):
    tx.run(query)

with open('new_data.txt', 'r') as file:
    # data = json.load(file)
    query = file.read()

with driver.session() as session:
    session.execute_write(add_complex_data, query)

print("Complex data added successfully!")

driver.close()