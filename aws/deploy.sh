#!/usr/bin/env bash

# ------------------------------------------------------------------
#  One-command CloudFormation launcher for the stack
#  ./deploy.sh  [stack-name]  [region]
#  Defaults: stack-name = intercountnet , region = us-east-1
# ------------------------------------------------------------------

set -euo pipefail

STACK_NAME=${1:-intercountnet}
REGION=${2:-us-east-1}
TEMPLATE_FILE="aws/intercountnet-aws-infra.yaml"

# ------------- fill your defaults ------------------------------
KEY_PAIR="my-key"            # must exist in the region
GITHUB_REPO="https://github.com/PSswathi/intercountnet-intersection-vehicle-counting-using-computer-vision.git"
S3_BUCKET="intercountnet-models"
INSTANCE_TYPE="t3.micro"
# ---------------------------------------------------------------

echo "☁️  Creating CloudFormation stack '$STACK_NAME' in $REGION ..."
aws cloudformation create-stack \
  --stack-name "$STACK_NAME" \
  --template-body "file://${TEMPLATE_FILE}" \
  --parameters \
    ParameterKey=KeyName,ParameterValue="$KEY_PAIR" \
    ParameterKey=GithubRepo,ParameterValue="$GITHUB_REPO" \
    ParameterKey=S3Bucket,ParameterValue="$S3_BUCKET" \
    ParameterKey=InstanceType,ParameterValue="$INSTANCE_TYPE" \
  --capabilities CAPABILITY_IAM \
  --region "$REGION"

echo "⏳  Waiting for stack to complete (~3 min) ..."
aws cloudformation wait stack-create-complete \
  --stack-name "$STACK_NAME" \
  --region "$REGION"

echo "✅  Stack created successfully."
aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs' \
  --region "$REGION"
