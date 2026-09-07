import uuid


def generate_request_id() -> str:
    """
    Generate a unique identifier for an API request.
    """
    return str(uuid.uuid4())