from crewai import Crew, Process, Task

from core.llm_config import get_llm

from agents.market_research import create_agent as create_market_agent
from agents.business_strategy import create_agent as create_strategy_agent
from agents.financial_analysis import create_agent as create_finance_agent
from agents.marketing_sales import create_agent as create_marketing_agent
from agents.operations import create_agent as create_operations_agent
from agents.risk_analysis import create_agent as create_risk_agent


def create_consulting_crew(
    business_name,
    business_description,
    target_market,
    location,
    budget,
    goals,
):
    llm = get_llm()

    market_agent = create_market_agent(llm)
    strategy_agent = create_strategy_agent(llm)
    finance_agent = create_finance_agent(llm)
    marketing_agent = create_marketing_agent(llm)
    operations_agent = create_operations_agent(llm)
    risk_agent = create_risk_agent(llm)

    market_task = Task(
        description=f"""
        Conduct market research for this business.

        Business Name: {business_name}
        Business Description: {business_description}
        Target Market: {target_market}
        Location: {location}

        Identify:
        - Market opportunities
        - Target customers
        - Competitors
        - Market trends
        - Customer needs
        """,
        expected_output="A detailed market research analysis.",
        agent=market_agent,
    )

    strategy_task = Task(
        description=f"""
        Develop a business strategy based on the market research.

        Business: {business_name}
        Goals: {goals}

        Recommend:
        - Business positioning
        - Competitive strategy
        - Business model
        - Growth strategy
        """,
        expected_output="A practical business strategy.",
        agent=strategy_agent,
        context=[market_task],
    )

    finance_task = Task(
        description=f"""
        Analyze the financial side of the business.

        Business: {business_name}
        Budget: {budget}

        Provide:
        - Estimated costs
        - Revenue opportunities
        - Pricing considerations
        - Profitability considerations
        - Financial risks
        """,
        expected_output="A financial analysis and recommendations.",
        agent=finance_agent,
        context=[market_task, strategy_task],
    )

    marketing_task = Task(
        description=f"""
        Create a marketing and sales plan for the business.

        Business: {business_name}
        Target Market: {target_market}

        Include:
        - Marketing channels
        - Customer acquisition
        - Branding
        - Sales strategy
        - Digital marketing ideas
        """,
        expected_output="A practical marketing and sales plan.",
        agent=marketing_agent,
        context=[market_task, strategy_task],
    )

    operations_task = Task(
        description=f"""
        Develop an operations plan for the business.

        Business: {business_name}
        Location: {location}

        Cover:
        - Daily operations
        - Required resources
        - Staffing
        - Technology
        - Operational workflow
        """,
        expected_output="A practical operations plan.",
        agent=operations_agent,
        context=[strategy_task, finance_task],
    )

    risk_task = Task(
        description=f"""
        Identify and analyze the major risks of this business.

        Business: {business_name}

        Consider:
        - Market risks
        - Financial risks
        - Operational risks
        - Competitive risks
        - Legal and regulatory considerations

        Provide mitigation strategies.
        """,
        expected_output="A risk analysis with mitigation strategies.",
        agent=risk_agent,
        context=[
            market_task,
            strategy_task,
            finance_task,
            marketing_task,
            operations_task,
        ],
    )

    crew = Crew(
        agents=[
            market_agent,
            strategy_agent,
            finance_agent,
            marketing_agent,
            operations_agent,
            risk_agent,
        ],
        tasks=[
            market_task,
            strategy_task,
            finance_task,
            marketing_task,
            operations_task,
            risk_task,
        ],
        process=Process.sequential,
        verbose=False,
    )

    return crew
