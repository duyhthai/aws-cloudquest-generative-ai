import os
import json
import boto3
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_community.retrievers import AmazonKendraRetriever
from langchain_community.chat_models import BedrockChat
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import traceback

# Set up the Bedrock client
bedrock = boto3.client('bedrock')

# Set up the Kendra client
kendra = boto3.client('kendra')

KENDRA_INDEX_ID = os.getenv('KENDRA_INDEX_ID')
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]


def lambda_handler(event, context):
    print(f"Event is: {event}")
    
    event_body = json.loads(event["body"])
    question = event_body["query"]
    prompt_template = event_body["prompt"]
    print(f"Query: {question}")
    print(f"Prompt: {prompt_template}")
    
    model_id = event_body["model_id"]
    temperature = event_body["temperature"]
    max_tokens = event_body["max_tokens"]

    response = ''
    status_code = 200
    print("model id : ", model_id)
    try:
        if model_id == 'amazon.titan-text-premier-v1:0':
            llm = get_titan_llm(model_id, temperature, max_tokens)
        else:
            llm = get_mistral_llm(model_id, temperature, max_tokens)

        # Initialize the Kendra loader
        retriever = AmazonKendraRetriever(
            kendra_client=kendra,
            index_id=KENDRA_INDEX_ID
        )

        prompt = PromptTemplate(
                template=prompt_template, input_variables=["context","question"]
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )
        
        #response = qa.invoke(question)
        response = qa(question, return_only_outputs=True)
        print(response)
        
        response_with_metadata = {
            "answer": response['result']
        }
        
            
        return {
            'statusCode': status_code,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps(response_with_metadata)
        }

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        stack_trace = traceback.format_exc()
        print(f"stack trace: {stack_trace}")
        print(f"error: {str(e)}")
        
        response = str(e)
        return {
            'statusCode': status_code,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({'error': response})
        }
        
def get_mistral_llm(model_id, temperature, max_tokens):
    model_kwargs = { 
        "max_tokens": max_tokens,
        "temperature": temperature, 
        "top_k": 50, 
        "top_p": 0.7
    }
    llm = BedrockChat(model_id=model_id, model_kwargs=model_kwargs) 
    return llm


# DIY section - add titan llm code block below (refer to ragFunction code for help)

