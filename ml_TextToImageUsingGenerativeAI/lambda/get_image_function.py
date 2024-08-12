import json
import boto3
import os
import logging
from operator import itemgetter

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create an S3 client
s3_client = boto3.client('s3')
s3 = boto3.resource('s3')
cf = boto3.client('cloudfront')


def lambda_handler(event, context):
    logger.info('Event: %s', json.dumps(event))

    # Retrieve latest image from S3
    bucket_name = os.environ['BUCKET_NAME']
    prefix = 'generated_images/'
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if 'Contents' in response:
        files = sorted(response['Contents'], key=itemgetter('LastModified'), reverse=True)
        latest_file = files[0]['Key']
        logger.info(f'Latest file in S3: {latest_file}')

        # Construct CloudFront URL
        distribution_id = cf.list_distributions()['DistributionList']['Items'][0]['Id']
        cf_domain_name = cf.get_distribution(Id=distribution_id)['Distribution']['DomainName']
        cf_domain_url = f'https://{cf_domain_name}/{latest_file}'

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': cf_domain_url
        }

    return {
        'statusCode': 404,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST'
        },
        'body': json.dumps({'error': 'No images found in S3'})
    }
    