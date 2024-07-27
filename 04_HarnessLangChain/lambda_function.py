# Import necessary libraries
import json
import boto3
import os
import re
import logging
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint, LLMContentHandler

# Configure logging for better traceability
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize a boto3 session to interact with AWS services
session = boto3.Session()
# Create a SageMaker Runtime client for invoking the endpoint
sagemaker_runtime_client = session.client("sagemaker-runtime")

# Retrieve the SageMaker endpoint name from environment variables
instruct_endpoint_name = os.environ["ENDPOINT_NAME"]

def lambda_handler(event, context):
    """
    AWS Lambda function handler to process input, invoke a SageMaker endpoint for NLP tasks,
    and return the processed output.
    
    :param event: The event dictionary containing the input data.
    :param context: Provides information about the invocation, function, and execution environment.
    """
    # Log the incoming event in JSON format for debugging
    logger.info('Event: %s', json.dumps(event))
    
    # Clean the event body: remove excess spaces and newline characters
    cleaned_body = re.sub(r'\s+', ' ', event.get('body', '')).replace('\n', '')
    cleaned_body = json.loads(cleaned_body)
    logger.info('Cleaned body: %s', cleaned_body)
    
    # SageMaker model parameters for the NLP task
    model_parameters = {
        "max_length": 200,
        "num_return_sequences": 1,
        "top_k": 250,
        "top_p": 0.95,
        "do_sample": False,
        "temperature": 1,
    }

    # Define a content handler for SageMaker endpoint interaction
    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"
        
        def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
            """Prepare the input payload for the SageMaker endpoint."""
            input_str = json.dumps({"inputs": prompt, **model_kwargs})
            return input_str.encode("utf-8")
        
        def transform_output(self, output: bytes) -> str:
            """Parse the output from the SageMaker endpoint."""
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json[0]["generated_text"]

    # Instantiate the content handler
    content_handler = ContentHandler()
    
    # Configure the SageMakerEndpoint instance with the necessary details
    sm_llm = SagemakerEndpoint(
        endpoint_name=instruct_endpoint_name,
        region_name='us-east-1',
        model_kwargs=model_parameters,
        content_handler=content_handler,
    )
    
    # Define the prompt template for the task
    prompt_template = "summarize the text {summarize_text}."

    # Create the prompt template instance
    prompt = PromptTemplate(
        input_variables=["summarize_text"],
        template=prompt_template,
    )

    # Initialize the LLMChain with the prompt template and SageMaker LLM
    llm_chain = LLMChain(prompt=prompt, llm=sm_llm)

    # Extract the task description from the cleaned event body and generate a response
    response = llm_chain.run(summarize_text=cleaned_body["inputs"])
    logger.info(response)
    
    # Prepare the response with a status code and headers for CORS compliance
    result = {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST'
        },
        'body': json.dumps(response)
    }
    
    # Log and return the result
    logger.info(result)
    return result
