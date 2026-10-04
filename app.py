
import streamlit as st

from core.crew_builder import create_consulting_crew
from core.report_builder import build_report


st.set_page_config(
    page_title="Business Consulting AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Business Consulting AI")
st.write(
    "A team of six AI business consultants powered by "
    "CrewAI, Groq, and GPT-OSS 120B."
)

with st.sidebar:
    st.subheader("System configuration")
    st.write("LLM provider: Groq")
    st.write("Model: openai/gpt-oss-120b")
    st.write("Agent framework: CrewAI")
    st.write("Specialist agents: 6")

with st.form("business_intake"):
    st.subheader("Business information")

    business_name = st.text_input(
        "Business or project name"
    )

    industry = st.text_input(
        "Industry",
        placeholder="For example, education or e-commerce",
    )

    location = st.text_input(
        "Target market or location",
        placeholder="For example, Lahore, Pakistan",
    )

    stage = st.selectbox(
        "Business stage",
        [
            "Business idea",
            "Planning to launch",
            "Recently launched",
            "Established business",
            "Business expansion",
        ],
    )

    goal = st.text_area("Main business goal")
    challenge = st.text_area("Main business challenge")

    budget = st.text_input(
        "Budget and currency",
        placeholder="For example, PKR 500,000 or undecided",
    )

    timeframe = st.selectbox(
        "Planning timeframe",
        ["3 months", "6 months", "12 months", "3 years"],
    )

    submitted = st.form_submit_button(
        "Generate consulting report",
        type="primary",
        use_container_width=True,
    )


if submitted:
    if not industry.strip() or not goal.strip():
        st.error("Please enter the industry and main business goal.")
        st.stop()

    name = business_name.strip() or "Unnamed Business"

    business_context = f"""
Business name: {name}
Industry: {industry.strip()}
Target market: {location.strip() or "Not specified"}
Business stage: {stage}
Main goal: {goal.strip()}
Main challenge: {challenge.strip() or "Not specified"}
Budget: {budget.strip() or "Not specified"}
Planning timeframe: {timeframe}

Instructions:
- Tailor all recommendations to these details.
- Distinguish estimates from verified facts.
- Do not fabricate research findings or statistics.
- Identify missing information and important assumptions.
"""

    try:
        with st.spinner(
            "CrewAI is coordinating your six consultants. "
            "This may take several minutes."
        ):
            crew = create_consulting_crew()
            result = crew.kickoff(
                inputs={"business_context": business_context}
            )

        report = build_report(name, result)

        st.success("Your consulting report is ready.")
        st.subheader("Consulting report")
        st.markdown(report)

        st.download_button(
            "Download report",
            data=report,
            file_name="business_consulting_report.md",
            mime="text/markdown",
        )

        with st.expander("View individual agent outputs"):
            task_outputs = getattr(result, "tasks_output", [])

            if task_outputs:
                for index, task_output in enumerate(task_outputs, start=1):
                    agent_name = (
                        getattr(task_output, "agent", None)
                        or f"Consulting task {index}"
                    )
                    st.markdown(f"### {agent_name}")
                    st.markdown(str(task_output.raw))
            else:
                st.markdown(str(result))

    except Exception as exc:
        st.error(
            "The consulting workflow failed. Check the Streamlit "
            "deployment logs and verify the Groq key, model access, "
            "dependencies, and API limits."
        )
        st.caption(f"Error type: {type(exc).__name__}")
