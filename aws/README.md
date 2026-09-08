# AWS Deployment

This directory contains the AWS configuration and deployment documentation for the Customer Churn Production ML Pipeline.

## Verified AWS Architecture

The currently verified deployment uses:

```text
Docker Image
     |
     v
Amazon ECR
     |
     v
Amazon ECS Fargate
     |
     +----> FastAPI inference service
     |
     +----> CloudWatch Logs