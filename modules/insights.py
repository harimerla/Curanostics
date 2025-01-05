def generate_structured_insights(summary):
    """
    Structure raw summaries into user-friendly sections.
    """
    # Example structuring
    insights = {
        "overview": summary[:150],  # Example: First 150 characters as overview
        "key_findings": summary[150:300],
        "recommendations": summary[300:]
    }
    return insights