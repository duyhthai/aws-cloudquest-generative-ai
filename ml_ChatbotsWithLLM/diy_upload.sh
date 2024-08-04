#!/usr/bin/bash
LAB_BUCKET=`aws s3 ls | grep lab-code | awk '{ print $3 }'`
S3_PREFIX='s3://'
echo "Upload bucket is ${LAB_BUCKET}"
aws s3 cp ./diy_results.txt ${S3_PREFIX}${LAB_BUCKET}