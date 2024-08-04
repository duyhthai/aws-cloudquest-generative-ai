#!/bin/bash
echo "Preparing Cloud9 for project deploymnet"

sudo yum install -y jq

echo "Update node js to version 16"
source "$HOME/.nvm/nvm.sh"
nvm install 16
nvm use 16
node --version

echo "Set region"
export AWS_REGION=$(curl -s 169.254.169.254/latest/dynamic/instance-identity/document | jq -r '.region')
echo "AWS Region is $AWS_REGION"

echo "Build the backend code using sam build"
sam build

echo "Export S3 bucket name and Kendra index which are created as part of Startup CFN stack"
export S3BucketName=$(aws s3api list-buckets --query "Buckets[?contains(Name, 'kendra-docs')].Name" --output text)
export LabBucketName=$(aws s3api list-buckets --query "Buckets[?contains(Name, 'lab-code')].Name" --output text)
export KendraIndexID=$(aws kendra list-indices --query "IndexConfigurationSummaryItems[?contains(Name, 'kendra-index')].Id" --output text)
export LAMBDA_ROLE_ARN=$(aws iam  list-roles --query "Roles[?contains(RoleName, 'LambdaDeploymentRole')].Arn" --output text)


export SAMStackName="sam-bedrock-stack"
echo $SAMStackName

echo "Copy toml file and replace the parameters"

cp tools/samconf.toml samconfig.toml
# Replace values in .//samconfig.toml
sed -Ei "s|<KendraIndexId>|${KendraIndexID}|g" ./samconfig.toml
sed -Ei "s|<S3BucketName>|${S3BucketName}|g" ./samconfig.toml
sed -Ei "s|<LabBucketName>|${LabBucketName}|g" ./samconfig.toml
sed -Ei "s|<SAMStackName>|${SAMStackName}|g" ./samconfig.toml
sed -Ei "s|<AWS_REGION>|${AWS_REGION}|g" ./samconfig.toml
sed -Ei "s|<LambdaRole>|${LAMBDA_ROLE_ARN}|g" samconfig.toml

echo "Deploy app with sam deploy"
sam deploy

echo "Export few more parameters"
export BedrockApiUrl=$(aws cloudformation describe-stacks --stack-name ${SAMStackName} --query "Stacks[0].Outputs[?OutputKey=='BedrockApiUrl'].OutputValue" --output text)
export UserPoolId=$(aws cognito-idp list-user-pools --query "UserPools[?contains(Name, 'ChatbotUserPool')].Id"  --max-results 1 --output text)
export UserPoolClientId=$(aws cognito-idp list-user-pool-clients --user-pool-id ${UserPoolId}  --query "UserPoolClients[?contains(ClientName, 'ChatbotUserPoolClient')].ClientId"  --output text)

echo "Gateway endpoint: $BedrockApiUrl"
echo "Cognito user pool id: $UserPoolId"
echo "Cognito client id: $UserPoolClientId"

# Replace values in ./backend/samconfig.toml
sed -Ei "s|<ApiGatewayUrl>|${BedrockApiUrl}|g" ./frontend/src/main.js
sed -Ei "s|<CognitoUserPoolId>|${UserPoolId}|g" ./frontend/src/main.js
sed -Ei "s|<UserPoolClientId>|${UserPoolClientId}|g" ./frontend/src/main.js

#Install Ampliyfy and build frontend
echo "Install Ampliyfy and build frontend"
cd ~/environment/serverless-chatbot-code/frontend
npm i -S @vue/cli-service
npm i -g @aws-amplify/cli
npm install
npm run build
cp ~/.aws/credentials ~/.aws/config

#Amplify initialization
echo "Amplify initialization"
mv dist build
amplify init --yes


echo "Add hosting, hit enter key if it prompts for action, use default"
amplify add hosting parameters.json

echo "Publish the amplify project"
amplify publish --yes

