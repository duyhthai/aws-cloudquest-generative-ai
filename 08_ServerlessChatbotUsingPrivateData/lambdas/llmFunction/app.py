import boto3
import json
import json
import traceback


region = boto3.session.Session().region_name

def lambda_handler(event, context):
    print(f"Event is: {event}")
    event_body = json.loads(event["body"])
    prompt = event_body["query"]
    temperature = event_body["temperature"]
    max_tokens = event_body["max_tokens"]
    model_id = event_body["model_id"]
    
    response = ''
    status_code = 200
    
    try:
        if model_id == 'mistral.mistral-7b-instruct-v0:2':
            response = invoke_mistral_7b(model_id, prompt, temperature, max_tokens)
        else:
            response = invoke_titan(model_id, prompt, temperature, max_tokens)
            
        return {
            'statusCode': status_code,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({'answer': response})
        }
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        stack_trace = traceback.format_exc()
        print(stack_trace)
        return {
            'statusCode': status_code,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({'error': str(e)})
        }


def invoke_mistral_7b(model_id, prompt, temperature, max_tokens):
    try:
        instruction = f"<s>[INST] {prompt} [/INST]"
        bedrock_runtime_client = boto3.client(service_name="bedrock-runtime", region_name=region)

        body = {
            "prompt": instruction,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = bedrock_runtime_client.invoke_model(
            modelId=model_id, body=json.dumps(body)
        )
        response_body = json.loads(response["body"].read())
        outputs = response_body.get("outputs")
        print(f"response: {outputs}")

        completions = [output["text"] for output in outputs]
        return completions[0]
    except Exception as e:
        raise
        
def invoke_titan(model_id, prompt, temperature, max_tokens):
    try:
        model_id="amazon.titan-text-premier-v1:0"
        bedrock_runtime_client = boto3.client(service_name="bedrock-runtime", region_name=region)
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "stopSequences": ["User:"],
                "temperature": temperature,
                "topP": 0.9,
            },
        }

        response = bedrock_runtime_client.invoke_model(
            modelId=model_id, body=json.dumps(body)
        )

        response_body = json.loads(response["body"].read())
        outputs = response_body.get("results")
        print(f"response: {outputs}")
        outputText = outputs[0].get("outputText")
        return outputText
    except Exception as e:
        raise
