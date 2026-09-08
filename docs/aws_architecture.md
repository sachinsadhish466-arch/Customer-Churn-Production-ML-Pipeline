\# AWS Deployment Architecture



\## Overview



The Customer Churn Production ML Pipeline is designed with an

AWS-oriented deployment architecture for containerized machine

learning inference.



The current application is fully containerized using Docker and

exposes a FastAPI inference service.



The AWS architecture described below represents the intended

deployment design.



Unless explicitly stated otherwise, AWS components in this

document are architectural targets and should not be interpreted

as a live production deployment.



\---



\# Architecture



```text

&#x20;                        AWS CLOUD

&#x20;                           |

&#x20;                   +-------v--------+

&#x20;                   |  API Gateway   |

&#x20;                   | HTTP Endpoint  |

&#x20;                   +-------+--------+

&#x20;                           |

&#x20;                   +-------v--------+

&#x20;                   |  ECS Fargate   |

&#x20;                   |    FastAPI     |

&#x20;                   | Dockerized API |

&#x20;                   +-------+--------+

&#x20;                           |

&#x20;            +--------------+--------------+

&#x20;            |              |              |

&#x20;      +-----v-----+   +----v-----+   +----v-------+

&#x20;      |    ECR    |   |    S3    |   | CloudWatch |

&#x20;      | Container |   | Data \&   |   | Logs \&     |

&#x20;      | Registry  |   | Artifacts|   | Monitoring |

&#x20;      +-----------+   +----------+   +------------+

