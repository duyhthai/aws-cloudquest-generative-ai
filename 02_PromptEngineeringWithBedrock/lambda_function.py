#!/usr/bin/env python3
import boto3
import json
from datetime import datetime


def lambda_handler(event, context):
    print(event, context)

    body = event['body'] 
    path = event['path']
    method = event['httpMethod']
    prompt = json.loads(body)['prompt']

    if path == "/invokemodel" and method == "POST":
        model_id = 'amazon.titan-text-premier-v1:0'
        model_response = call_bedrock(model_id, prompt)
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(model_response)
        }
    else:
        return {
            'statusCode': 404,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'message': 'Not Found'})
        }


def call_bedrock(modelId, prompt_data):
    bedrock_runtime = boto3.client('bedrock-runtime')
    
    prompt_template = """
    User Metadata:
    Name: John Doe
    Age: 35
    Location: New York, USA
    Interests: Technology, Artificial Intelligence, Music
    Occupation: Software Engineer
    
    Question: {prompt_data}
    """.format(prompt_data=prompt_data)
    
    body = json.dumps({
        "inputText": prompt_template,
        "textGenerationConfig": {
            "maxTokenCount": 1000,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 0.9
        }
    })
    print("bedrock-input:",body)

    accept = 'application/json'
    contentType = 'application/json'

    before = datetime.now()
    response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    latency = (datetime.now() - before).seconds
    response_body = json.loads(response.get('body').read())
    response = response_body.get('results')[0].get('outputText')
    
    return {
        'latency': str(latency),
        'response': response
    }
