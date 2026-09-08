\# AWS Deployment Configuration



This directory contains the AWS-oriented infrastructure configuration for the Customer Churn Production ML Pipeline.



\## Architecture



```text

GitHub

&#x20;  |

&#x20;  v

GitHub Actions

&#x20;  |

&#x20;  v

Amazon ECR

&#x20;  |

&#x20;  | Docker Image

&#x20;  v

Amazon ECS Fargate

&#x20;  |

&#x20;  | FastAPI

&#x20;  v

API Gateway

&#x20;  |

&#x20;  v

End Users





Supporting Services:



Amazon S3

&#x20;  |

&#x20;  +-- Model artifacts

&#x20;  +-- Monitoring artifacts

&#x20;  +-- Reports



Amazon CloudWatch

&#x20;  |

&#x20;  +-- Application logs

&#x20;  +-- Container logs

&#x20;  +-- Operational monitoring

