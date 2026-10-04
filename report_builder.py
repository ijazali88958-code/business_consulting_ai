def build_report(result):
    """
    Convert the CrewAI result into a clean report string.
    """

    if result is None:
        return "No report was generated."

    return str(result)
