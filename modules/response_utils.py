def create_response(insights):
    """
    Create structured JSON response for the API.
    """
    return {
        "status": "success",
        "data": insights
    }