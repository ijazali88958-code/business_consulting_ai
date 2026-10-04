
from crewai import Crew, Process, Task

from core.llm_config import get_llm

from agents.market_research import create_agent as create_market_agent
from agents.business_strategy import create_agent as create_strategy_agent
from agents.financial_analysis import create_agent as create_finance_agent
from agents.marketing_sales import create_agent as create_marketing_agent
from agents.operations import create_agent as create_operations_agent
from agents.risk_analysis import create_agent as create_risk_agent


def create_consulting_crew():
    llm = get_llm()

    market = create_market_agent(llm)
    strategy = create_strategy_agent(llm)
    finance = create_finance_agent(llm)
    marketing = create_marketing_agent(llm)
    operations = create_operations_agent(llm)
    risk = create_risk_agent(llm)

    market_task = Task(
        description=(
            "Analyze customers, competitors, opportunities, "
            "and barriers for this business:\n{business_context}"
        ),
        expected_output="Structured market research with assumptions.",
        agent=market,
    )

    strategy_task = Task(
        description=(
            "Develop business goals, positioning, business model, "
            "and growth strategy for:\n{business_context}"
        ),
        expected_output="A practical business strategy.",
        agent=strategy,
        context=[market_task],
    )

    finance_task = Task(
        description=(
            "Analyze startup costs, expenses, revenue assumptions, "
            "cash flow, and break-even planning for:\n{business_context}. "
            "Clearly label all estimates."
        ),
        expected_output="Financial planning analysis and assumptions.",
        agent=finance,
        context=[strategy_task],
    )

    marketing_task = Task(
        description=(
            "Create a marketing and sales plan for:\n{business_context}"
        ),
        expected_output="Marketing channels, sales actions, and KPIs.",
        agent=marketing,
        context=[market_task, strategy_task],
    )

    operations_task = Task(
        description=(
            "Develop an operations plan for:\n{business_context}"
        ),
        expected_output="Workflows, resources, and implementation steps.",
        agent=operations,
        context=[strategy_task, finance_task],
    )

    risk_task = Task(
        description=(
            "Analyze business risks, impacts, warning indicators, "
            "and mitigation strategies for:\n{business_context}. "
            "Do not invent facts."
        ),
        expected_output="A structured business risk analysis.",
        agent=risk,
        context=[
            market_task,
            strategy_task,
            finance_task,
            marketing_task,
            operations_task,
        ],
    )

    return Crew(
        agents=[market, strategy, finance, marketing, operations, risk],
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
