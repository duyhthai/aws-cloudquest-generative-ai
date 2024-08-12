import json
import boto3
import os
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a Lambda client
lambda_client = boto3.client('lambda')

def lambda_handler(event, context):
    logger.info('Event: %s', json.dumps(event))

    # Asynchronously invoke another Lambda function to handle processing
    processing_lambda_name = os.environ["PROCESSING_LAMBDA_NAME"]
    lambda_client.invoke(
        FunctionName=processing_lambda_name,
        InvocationType='Event',  # Asynchronous invocation
        Payload=json.dumps(event)
    )

    # Return the result with status code 200 and the necessary headers
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST'
        },
        'body': json.dumps({'message': 'Request received, processing started'})
    }