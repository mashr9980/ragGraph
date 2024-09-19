import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from tenacity import retry, stop_after_attempt, wait_random_exponential
import json
from langchain.schema import BaseMessage

# Import from previous steps
from Data_Ingestion_and_Knowledge_Graph_Creation import Neo4jConnector
from Query_Engine_and_Multi_Agent_Architecture import Orchestrator

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMIntegration:
    def __init__(self):
        self.llm = self._setup_llm()
        self.response_parser = self._setup_response_parser()
        self.prompt_template = self._setup_prompt_template()

    def _setup_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_type="azure",
            temperature=0.7
        )

    def _setup_response_parser(self) -> StructuredOutputParser:
        response_schemas = [
            ResponseSchema(name="summary", description="A brief summary of the response"),
            ResponseSchema(name="detailed_answer", description="The detailed answer to the query"),
            ResponseSchema(name="confidence", description="Confidence level in the response (low, medium, high)"),
            ResponseSchema(name="follow_up_questions", description="List of potential follow-up questions")
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)

    def _setup_prompt_template(self) -> PromptTemplate:
        template = """You are an AI assistant for a knowledge graph system about microservices, tasks, and teams.
        Answer the following query based on the provided context. Be concise yet informative.

        Query: {query}

        Context: {context}

        {format_instructions}

        Response:"""
        return PromptTemplate(
            template=template,
            input_variables=["query", "context"],
            partial_variables={"format_instructions": self.response_parser.get_format_instructions()}
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def generate_response(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = self.prompt_template.format(query=query, context=json.dumps(context))
            response = self.llm.predict(prompt)
            
            response_text = self._extract_response_text(response)
            parsed_response = self._parse_llm_response(response_text)
            
            logger.info(f"Generated response for query: {query}")
            return parsed_response

        except Exception as e:
            logger.error(f"Error generating response in LLMIntegration: {str(e)}")
            return {
                "summary": "Unable to generate response",
                "detailed_answer": f"An error occurred: {str(e)}",
                "confidence": "low",
                "follow_up_questions": []
            }

    def _extract_response_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        elif isinstance(response, BaseMessage):
            return response.content
        elif hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'content'):
            return response.content
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        try:
            return self.response_parser.parse(response)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                "summary": "Error in response parsing",
                "detailed_answer": response,
                "confidence": "low",
                "follow_up_questions": []
            }

class ResponseGenerator:
    def __init__(self, llm_integration: LLMIntegration):
        self.llm_integration = llm_integration

    def process_query(self, query: str, orchestrator_result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            context = orchestrator_result.get('context', {})
            enhanced_response = self.llm_integration.generate_response(query, context)
            
            return {
                "orchestrator_result": orchestrator_result,
                "enhanced_response": enhanced_response
            }
        except Exception as e:
            logger.error(f"Error processing query in ResponseGenerator: {str(e)}")
            return {
                "error": f"An error occurred while processing the query: {str(e)}",
                "orchestrator_result": orchestrator_result,
                "enhanced_response": {
                    "summary": "Error in response generation",
                    "detailed_answer": "The system encountered an error while generating a response.",
                    "confidence": "low",
                    "follow_up_questions": []
                }
            }

def main():
    neo4j_connector = Neo4jConnector()
    try:
        neo4j_connector.connect()
        
        orchestrator = Orchestrator(neo4j_connector)
        orchestrator.initialize()

        llm_integration = LLMIntegration()
        response_generator = ResponseGenerator(llm_integration)

        # Example usage
        queries = [
            "What are the dependencies of the PaymentService?",
            "Who is responsible for the UserAuthenticationService?",
            "What are the most critical open tasks across all microservices?"
        ]

        for query in queries:
            print(f"\nProcessing query: {query}")
            orchestrator_result = orchestrator.process_query(query)
            final_result = response_generator.process_query(query, orchestrator_result)
            print(f"Result: {final_result}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        neo4j_connector.close()

if __name__ == "__main__":
    main()