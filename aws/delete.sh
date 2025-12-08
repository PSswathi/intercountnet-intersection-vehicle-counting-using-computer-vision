#!/usr/bin/env bash

REGION=${2:-us-east-1}
STACK_NAME=${1:-intercountnet}
aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"
echo "🧹  Stack deletion started."
