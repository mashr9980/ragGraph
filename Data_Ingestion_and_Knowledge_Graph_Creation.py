import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jConnector:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = None

    def connect(self) -> None:
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {str(e)}")
            raise

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def get_driver(self) -> Driver:
        if not self.driver:
            self.connect()
        return self.driver

class DataIngestion:
    def __init__(self, neo4j_connector: Neo4jConnector):
        self.neo4j_connector = neo4j_connector

    def ingest_data(self, data_source: str) -> None:
        driver = self.neo4j_connector.get_driver()
        with driver.session() as session:
            if data_source == "microservices":
                self._ingest_microservices(session)
            elif data_source == "tasks":
                self._ingest_tasks(session)
            elif data_source == "teams":
                self._ingest_teams(session)
            else:
                logger.warning(f"Unknown data source: {data_source}")

    def _ingest_microservices(self, session) -> None:
        query = """
        MERGE (m:Microservice {name: $name})
        SET m.technology = $technology
        """
        microservices_data = self._load_json_data("microservices.json")
        for microservice in microservices_data:
            self._execute_query(session, query, microservice)

    def _ingest_tasks(self, session) -> None:
        query = """
        MERGE (t:Task {name: $name})
        SET t.status = $status, t.description = $description
        WITH t
        MATCH (m:Microservice {name: $microservice})
        MERGE (t)-[:RELATED_TO]->(m)
        """
        tasks_data = self._load_json_data("tasks.json")
        for task in tasks_data:
            self._execute_query(session, query, task)

    def _ingest_teams(self, session) -> None:
        query = """
        MERGE (team:Team {name: $name})
        WITH team
        UNWIND $members as member
        MERGE (p:Person {name: member})
        MERGE (p)-[:MEMBER_OF]->(team)
        WITH team
        MATCH (m:Microservice {name: $responsible_for})
        MERGE (team)-[:RESPONSIBLE_FOR]->(m)
        """
        teams_data = self._load_json_data("teams.json")
        for team in teams_data:
            self._execute_query(session, query, team)

    @staticmethod
    def _load_json_data(filename: str) -> List[Dict[str, Any]]:
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {filename}")
            return []

    @staticmethod
    def _execute_query(session, query: str, params: Dict[str, Any]) -> None:
        try:
            session.run(query, params)
            logger.info(f"Successfully ingested data: {params}")
        except Neo4jError as e:
            logger.error(f"Neo4j error during data ingestion: {str(e)}")

class KnowledgeGraphCreator:
    def __init__(self, neo4j_connector: Neo4jConnector):
        self.neo4j_connector = neo4j_connector

    def create_schema(self) -> None:
        driver = self.neo4j_connector.get_driver()
        with driver.session() as session:
            self._create_constraints(session)
            self._create_indexes(session)

    def _create_constraints(self, session) -> None:
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Microservice) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (team:Team) REQUIRE team.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE"
        ]
        for constraint in constraints:
            self._execute_query(session, constraint)

    def _create_indexes(self, session) -> None:
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (m:Microservice) ON (m.technology)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Task) ON (t.status)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)"
        ]
        for index in indexes:
            self._execute_query(session, index)

    @staticmethod
    def _execute_query(session, query: str) -> None:
        try:
            session.run(query)
            logger.info(f"Successfully executed query: {query}")
        except Neo4jError as e:
            logger.error(f"Neo4j error during query execution: {str(e)}")

class VectorIndexCreator:
    def __init__(self, neo4j_connector: Neo4jConnector):
        self.neo4j_connector = neo4j_connector
        self.embedding_model = "text-embedding-ada-002"
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def create_embedding(self, text: str) -> List[float]:
        try:
            response = openai.Embedding.create(input=text, model=self.embedding_model)
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    def create_vector_index(self) -> None:
        driver = self.neo4j_connector.get_driver()
        with driver.session() as session:
            try:
                query = """
                CALL db.index.vector.createNodeIndex(
                    'task_vector_index',
                    'Task',
                    'embedding',
                    1536,
                    'cosine'
                )
                """
                session.run(query)
                logger.info("Successfully created vector index")
            except Neo4jError as e:
                if "An equivalent index already exists" in str(e):
                    logger.info("Vector index already exists")
                else:
                    logger.error(f"Error creating vector index: {str(e)}")
                    raise

    def update_task_embeddings(self) -> None:
        driver = self.neo4j_connector.get_driver()
        with driver.session() as session:
            query = """
            MATCH (t:Task)
            WHERE t.embedding IS NULL
            RETURN t.name AS name, t.description AS description
            """
            result = session.run(query)
            tasks = [record for record in result]

            for task in tasks:
                text = f"{task['name']} {task['description']}"
                embedding = self.create_embedding(text)
                update_query = """
                MATCH (t:Task {name: $name})
                SET t.embedding = $embedding
                """
                session.run(update_query, {"name": task["name"], "embedding": embedding})
                logger.info(f"Updated embedding for task: {task['name']}")

def main():
    neo4j_connector = Neo4jConnector()
    try:
        neo4j_connector.connect()

        # Data Ingestion
        data_ingestion = DataIngestion(neo4j_connector)
        data_sources = ["microservices", "tasks", "teams"]
        for source in data_sources:
            data_ingestion.ingest_data(source)

        # Knowledge Graph Creation
        kg_creator = KnowledgeGraphCreator(neo4j_connector)
        kg_creator.create_schema()

        # Vector Index Creation
        vector_index_creator = VectorIndexCreator(neo4j_connector)
        vector_index_creator.create_vector_index()
        vector_index_creator.update_task_embeddings()

        logger.info("Data ingestion and knowledge graph creation completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        neo4j_connector.close()

if __name__ == "__main__":
    main()