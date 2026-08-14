import os
import uuid
from langsmith import Client
from dotenv import load_dotenv
from evals.eval_functions import evaluate_success_criteria, evaluate_no_assumptions
from research_agent_scope import graph as scope

load_dotenv()
langsmith_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

dataset_name = "deep_research_scoping"

def target_func(inputs: dict):
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    return scope.invoke(inputs, config=config)

langsmith_client.evaluate(
    target_func,
    data=dataset_name,
    evaluators=[evaluate_success_criteria, evaluate_no_assumptions],
    experiment_prefix="Deep Research Scoping",
)