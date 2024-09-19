import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

# Import from previous steps
from Data_Ingestion_and_Knowledge_Graph_Creation import Neo4jConnector
from Query_Engine_and_Multi_Agent_Architecture import Orchestrator
from LLM_Integration_and_Response_Generation import ResponseGenerator, LLMIntegration

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QueryContext:
    query: str
    start_time: float
    end_time: float = None
    error: str = None

class RobustQueryEngine:
    def __init__(self, orchestrator: Orchestrator, response_generator: ResponseGenerator):
        self.orchestrator = orchestrator
        self.response_generator = response_generator

    def execute_query(self, query: str) -> Dict[str, Any]:
        try:
            orchestrator_result = self.orchestrator.process_query(query)
            final_result = self.response_generator.process_query(query, orchestrator_result)
            return final_result
        except Exception as e:
            logger.error(f"Error executing query in RobustQueryEngine: {str(e)}")
            return {
                "error": f"An error occurred while processing the query: {str(e)}",
                "enhanced_response": {
                    "summary": "Error in query processing",
                    "detailed_answer": "The system encountered an error while processing your query.",
                    "confidence": "low",
                    "follow_up_questions": []
                }
            }

    def _log_query_metrics(self, context: QueryContext):
        duration = context.end_time - context.start_time
        logger.info(f"Query: {context.query}, Duration: {duration:.2f}s, Error: {context.error or 'None'}")

class LoadBalancer:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, num_workers: int = 3):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.query_engines = []
        self.neo4j_connectors = []

    def initialize(self):
        for _ in range(self.executor._max_workers):
            neo4j_connector = Neo4jConnector()
            neo4j_connector.connect()
            self.neo4j_connectors.append(neo4j_connector)
            orchestrator = Orchestrator(neo4j_connector)
            orchestrator.initialize()
            llm_integration = LLMIntegration()
            response_generator = ResponseGenerator(llm_integration)
            query_engine = RobustQueryEngine(orchestrator, response_generator)
            self.query_engines.append(query_engine)

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            query_engine = self.query_engines[hash(query) % len(self.query_engines)]
            return query_engine.execute_query(query)
        except Exception as e:
            logger.error(f"Error in LoadBalancer process_query: {str(e)}")
            return {"error": f"LoadBalancer error: {str(e)}"}

    def shutdown(self):
        self.executor.shutdown()
        for connector in self.neo4j_connectors:
            try:
                connector.close()
            except Exception as e:
                logger.error(f"Error closing Neo4j connector: {str(e)}")

@lru_cache(maxsize=1000)
def cached_query_execution(query: str, load_balancer: LoadBalancer) -> Dict[str, Any]:
    return load_balancer.process_query(query)

def main():
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    load_balancer = LoadBalancer(neo4j_uri, neo4j_user, neo4j_password)
    load_balancer.initialize()

    try:
        queries = [
            "What are the dependencies of the PaymentService?",
            "Who is responsible for the UserAuthenticationService?",
            "What are the most critical open tasks across all microservices?",
            "List all microservices and their team leads",
            "What is the current status of the DatabaseOptimizationTask?",
        ]

        for query in queries:
            print(f"\nProcessing query: {query}")
            result = cached_query_execution(query, load_balancer)
            print(f"Result: {result}")

        # Simulate concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_results = [executor.submit(cached_query_execution, query, load_balancer) for query in queries]
            for future in future_results:
                print(f"Concurrent result: {future.result()}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        load_balancer.shutdown()

if __name__ == "__main__":
    main()