
from crewai import Agent


def create_agent(llm):
    return Agent(
        role="Business Operations Consultant",
        goal=(
            "Recommend workflows, staffing, resources, "
            "quality control, and implementation steps."
        ),
        backstory=(
            "You are a business operations specialist who "
            "develops practical and cost-conscious processes."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
