#!/usr/bin/env bash
# ------------------------------------------------------------------
#  update.sh  – update the existing CloudFormation stack in-place
#  ./update.sh  [stack-name]  [region]
#  Defaults: stack-name = intercountnet , region = us-east-1
# ------------------------------------------------------------------

set -euo pipefail

STACK_NAME=${1:-intercountnet}
REGION=${2:-us-east-1}
TEMPLATE_FILE="aws/intercountnet-aws-infra.yaml"

# ------------- same defaults as deploy.sh -------------------------
KEY_PAIR="my-key"
GITHUB_REPO="https://github.com/PSswathi/intercountnet-intersection-vehicle-counting-using-computer-vision.git"
S3_BUCKET="intercountnet-models"
INSTANCE_TYPE="t3.medium"
# ---------------------------------------------------------------

echo "☁️  Updating CloudFormation stack '$STACK_NAME' in $REGION ..."
aws cloudformation update-stack \
  --stack-name "$STACK_NAME" \
  --template-body "file://${TEMPLATE_FILE}" \
  --parameters \
    ParameterKey=KeyName,ParameterValue="$KEY_PAIR" \
    ParameterKey=GithubRepo,ParameterValue="$GITHUB_REPO" \
    ParameterKey=S3Bucket,ParameterValue="$S3_BUCKET" \
    ParameterKey=InstanceType,ParameterValue="$INSTANCE_TYPE" \
  --capabilities CAPABILITY_IAM \
  --region "$REGION" \
  --disable-rollback

echo "⏳  Waiting for update to complete ..."
aws cloudformation wait stack-update-complete \
  --stack-name "$STACK_NAME" \
  --region "$REGION"

echo "✅  Stack updated successfully."
aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs' \
  --region "$REGION"
