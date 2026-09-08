# AWS Deployment Architecture

## Overview

The Customer Churn Production ML Pipeline uses a containerized AWS deployment architecture for machine learning inference.

The application is packaged using Docker and exposes a FastAPI inference service.

The **current verified portfolio deployment** uses Amazon ECR, Amazon ECS Fargate, IAM, and Amazon CloudWatch Logs.

This document distinguishes the verified deployment from potential future production enhancements.

---

## Current Verified AWS Architecture

```text
                         AWS CLOUD
                            |
                            |
                    +-------v--------+
                    |      ECR       |
                    | Container      |
                    | Image Registry |
                    +-------+--------+
                            |
                            | Docker Image
                            |
                    +-------v--------+
                    |  ECS Fargate   |
                    |                |
                    |    FastAPI     |
                    |  Dockerized API|
                    |                |
                    |    Port 8000   |
                    +-------+--------+
                            |
                +-----------+-----------+
                |                       |
                |                       |
        +-------v--------+      +-------v--------+
        |   CloudWatch  |      |   Prediction   |
        |     Logs      |      |   Monitoring   |
        |               |      |  & Drift API   |
        +----------------+      +----------------+