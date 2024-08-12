import json
import boto3
import os
import logging
from PIL import Image
import io
import base64
import uuid

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a SageMaker client
sagemaker_client = boto3.client("sagemaker-runtime")

# Create an S3 client
s3_client = boto3.client('s3')
s3 = boto3.resource('s3')
cf = boto3.client('cloudfront')

# Get the name of the S3 bucket with the prefix, lab-code.
bucket_name = os.environ['BUCKET_NAME']
s3_folder = 'generated_images/'

def create_and_upload_image(image_data):
    image = Image.new("RGB", (len(image_data[0]), len(image_data)))
    pixels = image.load()

    for i in range(image.width):
        for j in range(image.height):
            pixels[i, j] = tuple(image_data[j][i])

    # Convert to JPG and upload to S3
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    file_name = f'generated-image-{uuid.uuid4()}.jpg'
    s3_client.upload_fileobj(
        buffer,
        bucket_name,
        f'{s3_folder}{file_name}',
        ExtraArgs={'ContentType': 'image/jpeg'}
    )

    return f'{s3_folder}{file_name}'


def lambda_handler(event, context):
    logger.info('Event: %s', json.dumps(event))
    
    body_content = json.loads(event['body'])
    cleaned_body = json.dumps(body_content, separators=(',', ':'))
    logger.info('Cleaned body: %s', cleaned_body)

    encoded_payload = cleaned_body.encode("utf-8")

    response = sagemaker_client.invoke_endpoint(
        EndpointName=os.environ["ENDPOINT_NAME"], 
        ContentType="application/json", 
        Body=encoded_payload
    )

    result = json.loads(response["Body"].read().decode())
    # If generated image is detected, save the file in the S3 bucket.

    if "generated_images" in result:
        s3_image_path = create_and_upload_image(result["generated_images"][0])
        
        logger.info(s3_image_path)

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Image processed and uploaded'})
        }

    return {
        'statusCode': 400,
        'body': json.dumps({'error': 'Response not in expected format'})
    }
