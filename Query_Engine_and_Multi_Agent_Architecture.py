import os
import logging
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import AzureChatOpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import json
import faiss

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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

    def get_driver(self):
        if not self.driver:
            self.connect()
        return self.driver

class LargeResultManager:
    def __init__(self, max_items: int = 1000, summary_length: int = 200):
        self.max_items = max_items
        self.summary_length = summary_length

    def summarize_result(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, list):
            return self._summarize_list(result)
        elif isinstance(result, dict):
            return self._summarize_dict(result)
        else:
            return {"summary": str(result)[:self.summary_length], "type": str(type(result))}

    def _summarize_list(self, result: List[Any]) -> Dict[str, Any]:
        summary = {
            "type": "list",
            "total_items": len(result),
            "sample": result[:min(5, len(result))],
            "truncated": len(result) > self.max_items
        }
        if len(result) > 0:
            if isinstance(result[0], dict):
                keys = set().union(*(d.keys() for d in result[:100]))  # Sample first 100 items
                summary["common_keys"] = list(keys)
            summary["value_types"] = Counter(type(item).__name__ for item in result[:100])
        return summary

    def _summarize_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "dict",
            "keys": list(result.keys()),
            "sample": {k: str(v)[:50] for k, v in list(result.items())[:5]},
            "total_keys": len(result)
        }

class EmbeddingManager:
    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.id_to_text = {}
        self.current_id = 0

    def add_embedding(self, text: str, embedding: List[float]):
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding)}")
        self.index.add(np.array([embedding], dtype=np.float32))
        self.id_to_text[self.current_id] = text
        self.current_id += 1

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.embedding_dim}, got {len(query_embedding)}")
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 indicates no match found
                results.append({
                    "text": self.id_to_text[idx],
                    "distance": float(dist)
                })
        return results

class IntentEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.intents = {
            "cypher_query": ["database", "query", "graph", "relationship", "node", "find", "match", "return"],
            "embedding": ["similar", "vector", "embedding", "semantic", "closest", "related"],
            "assessment": ["security", "risk", "vulnerability", "assess", "evaluate", "audit"]
        }
        self.intent_vectors = self._prepare_intent_vectors()

    def _prepare_intent_vectors(self):
        intent_texts = [" ".join(keywords) for keywords in self.intents.values()]
        return self.vectorizer.fit_transform(intent_texts)

    def analyze_intent(self, query: str) -> str:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.intent_vectors)
        max_similarity_index = np.argmax(similarities)
        return list(self.intents.keys())[max_similarity_index]

class CypherQueryAgent:
    def __init__(self, neo4j_connector: Neo4jConnector):
        self.neo4j_connector = neo4j_connector
        self.graph = None
        self.qa_chain = None
        self.max_result_size = 1000
        self.context = []
        self.large_result_manager = LargeResultManager()

    def initialize(self):
        try:
            self.graph = Neo4jGraph(
                url=self.neo4j_connector.uri,
                username=self.neo4j_connector.user,
                password=self.neo4j_connector.password
            )
            
            self.graph.refresh_schema()

            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )

            self.qa_chain = GraphCypherQAChain.from_llm(
                self.llm,
                graph=self.graph,
                verbose=True
            )

            logger.info("Cypher Query Agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Cypher Query Agent: {str(e)}")
            raise

    def execute_query(self, query: str) -> Dict[str, Any]:
        try:
            context_query = self._add_context_to_query(query)
            
            if "RETURN" in context_query.upper() and "LIMIT" not in context_query.upper():
                context_query += f" LIMIT {self.max_result_size}"

            result = self.qa_chain(context_query)
            
            if isinstance(result, dict) and 'result' in result:
                result['result'] = self.large_result_manager.summarize_result(result['result'])

            self._update_context(query, result)

            logger.info(f"Query executed successfully: {context_query}")
            return result
        except Exception as e:
            error_message = f"Error executing query: {str(e)}"
            logger.error(error_message)
            return self._handle_query_error(query, error_message)

    def _add_context_to_query(self, query: str) -> str:
        if self.context:
            return f"{self.context[-1]['query']} AND {query}"
        return query

    def _update_context(self, query: str, result: Dict[str, Any]):
        self.context.append({"query": query, "result": result})
        self.context = self.context[-5:]  # Keep only the last 5 queries in context

    def _handle_query_error(self, query: str, error_message: str) -> Dict[str, Any]:
        return {
            "error": error_message,
            "query": query,
            "suggestion": "Please check your query syntax or try a simpler query.",
            "context": self.context
        }

class EmbeddingAgent:
    def __init__(self):
        self.model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        self.embedding_manager = EmbeddingManager()

    def generate_embedding(self, text: str) -> List[float]:
        # This is a placeholder. In a real implementation, you'd use a proper embedding model.
        return self.model.encode(text)

    def find_similar(self, query: str, corpus: List[str]) -> List[Dict[str, Any]]:
        query_embedding = self.generate_embedding(query)
        
        # Add corpus to embedding manager if not already present
        for item in corpus:
            if item not in self.embedding_manager.id_to_text.values():
                item_embedding = self.generate_embedding(item)
                self.embedding_manager.add_embedding(item, item_embedding)
        
        return self.embedding_manager.search(query_embedding)

class AssessmentAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

    def generate_assessment(self, asset: str) -> Dict[str, Any]:
        prompt = f"""Perform a comprehensive security assessment for the following asset: {asset}
        Consider the following aspects:
        1. Potential vulnerabilities
        2. Compliance with industry standards
        3. Access control mechanisms
        4. Data protection measures
        5. Incident response readiness
        
        Provide a detailed analysis with specific recommendations."""

        response = self.llm(prompt)
        
        # Parse the response into a structured format
        assessment = self._parse_assessment(response)
        
        return {"asset": asset, "assessment": assessment}

    def _parse_assessment(self, response: str) -> Dict[str, Any]:
        # This is a simple parser. In a real-world scenario, you might want to use
        # a more sophisticated parsing method or fine-tuned model for this task.
        sections = ["vulnerabilities", "compliance", "access_control", "data_protection", "incident_response"]
        assessment = {}
        
        for section in sections:
            pattern = f"{section.replace('_', ' ').title()}:(.*?)(?:$|{sections[sections.index(section)+1].replace('_', ' ').title() + ':' if sections.index(section) < len(sections)-1 else '$'})"
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                assessment[section] = match.group(1).strip()
            else:
                assessment[section] = "No specific information provided."
        
        return assessment

class MultiToolAgent:
    def __init__(self, cypher_agent: CypherQueryAgent, embedding_agent: EmbeddingAgent, assessment_agent: AssessmentAgent):
        self.cypher_agent = cypher_agent
        self.embedding_agent = embedding_agent
        self.assessment_agent = assessment_agent
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

    def execute_query(self, query: str) -> Dict[str, Any]:
        # Define the tools
        tools = [
            Tool(
                name="CypherQueryTool",
                func=self.cypher_agent.execute_query,
                description="Useful for querying the Neo4j graph database using Cypher queries."
            ),
            Tool(
                name="EmbeddingTool",
                func=self.embedding_agent.find_similar,
                description="Useful for finding similar items based on semantic meaning."
            ),
            Tool(
                name="AssessmentTool",
                func=self.assessment_agent.generate_assessment,
                description="Useful for generating security assessments for specific assets."
            )
        ]

        # Define the prompt template
        template = """You are an AI assistant tasked with answering queries about a microservices system.
        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: """

        # Set up the prompt
        prompt = StringPromptTemplate(
            template=template,
            tools=tools,
            input_variables=["input"],
            partial_variables={"tool_names": ", ".join([tool.name for tool in tools])}
        )

        # Set up the output parser
        class CustomOutputParser(AgentOutputParser):
            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                if "Final Answer:" in llm_output:
                    return AgentFinish(
                        return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                        log=llm_output,
                    )
                
                regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                if not match:
                    raise ValueError(f"Could not parse LLM output: `{llm_output}`")
                action = match.group(1).strip()
                action_input = match.group(2)
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

        output_parser = CustomOutputParser()

        # Set up the LLM chain
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        # Set up the agent
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in tools]
        )

        # Set up the agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

        # Execute the query
        result = agent_executor.run(query)

        return {"multi_tool_result": result}

class Orchestrator:
    def __init__(self, neo4j_connector: Neo4jConnector):
        self.neo4j_connector = neo4j_connector
        self.cypher_agent = CypherQueryAgent(neo4j_connector)
        self.embedding_agent = EmbeddingAgent()
        self.assessment_agent = AssessmentAgent()
        self.multi_tool_agent = MultiToolAgent(self.cypher_agent, self.embedding_agent, self.assessment_agent)
        self.intent_engine = IntentEngine()

    def initialize(self):
        self.cypher_agent.initialize()

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            intent = self.intent_engine.analyze_intent(query)
            logger.info(f"Detected intent: {intent}")

            if intent == "cypher_query":
                result = self.cypher_agent.execute_query(query)
            elif intent == "embedding":
                corpus = ["PaymentService", "UserAuthenticationService", "OrderProcessingService", "InventoryManagementService"]
                result = self.embedding_agent.find_similar(query, corpus)
            elif intent == "assessment":
                asset = query.split("for")[-1].strip() if "for" in query else query
                result = self.assessment_agent.generate_assessment(asset)
            else:
                # If the intent is unclear or the query seems complex, use the MultiToolAgent
                result = self.multi_tool_agent.execute_query(query)

            return {
                "intent": intent,
                "result": result,
                "context": self.cypher_agent.context
            }

        except Exception as e:
            logger.error(f"Error processing query in Orchestrator: {str(e)}")
            return self._handle_error(query, str(e))
        
    def _handle_error(self, query: str, error: str) -> Dict[str, Any]:
        logger.info(f"Attempting to recover from error: {error}")
        try:
            # Fallback to a simple Cypher query
            cypher_query = "MATCH (n) RETURN n.name AS name, labels(n) AS labels LIMIT 5"
            result = self.cypher_agent.execute_query(cypher_query)
            return {
                "error": f"An error occurred while processing the query: {error}",
                "fallback_result": result,
                "suggestion": "You might want to try rephrasing your query or breaking it into smaller parts.",
                "context": self.cypher_agent.context
            }
        except Exception as e:
            logger.error(f"Fallback query failed: {str(e)}")
            return {
                "error": f"Unable to process query: {query}. Error: {error}",
                "suggestion": "The system is currently experiencing issues. Please try again later or contact support.",
                "context": self.cypher_agent.context
            }

def main():
    neo4j_connector = Neo4jConnector()
    try:
        neo4j_connector.connect()
        
        orchestrator = Orchestrator(neo4j_connector)
        orchestrator.initialize()

        # Example usage
        queries = [
            "What are the dependencies of the PaymentService?",
            "Find similar microservices to the UserAuthenticationService",
            "Perform a security assessment for the DatabaseService",
            "What are the most critical open tasks across all microservices?",
            "Analyze the security of the PaymentService and suggest improvements based on its dependencies",
            "MATCH (n) RETURN n"  # This query might return a large result set
        ]

        for query in queries:
            print(f"\nProcessing query: {query}")
            result = orchestrator.process_query(query)
            print(f"Result: {result}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        neo4j_connector.close()

if __name__ == "__main__":
    main()