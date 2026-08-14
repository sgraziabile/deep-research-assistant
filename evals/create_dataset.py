import os
from langsmith import Client
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv(".env.local")

# Conversations and criteria for the dataset
conversation_1 = [
    HumanMessage(content="What's the best way to invest $50,000 for retirement?"),
    AIMessage(content="Could you please provide some additional information to tailor the investment advice for your $50,000 retirement goal? Specifically:\n Your current age or desired retirement age\n Your risk tolerance (low, medium, high)\n Any preferences for investment types (e.g., stocks, bonds, mutual funds, real estate)\n Whether you are investing through a tax-advantaged account (e.g., IRA, 401(k)) or a regular brokerage account\n This will help me provide more personalized and relevant suggestions."),
    HumanMessage(content="I'm 25 and I want to retire by 45. My risk tolerance is high right now but I think will decrease over time. I have heard that stocks and ETFs are a good choice, but I'm open to anything. And I already have a 401k, but this would just be through a regular brokerage account."),
]

conversation_2 = [
    HumanMessage(content="I am looking for an apartment in NYC, can you help me?"),
    AIMessage(content="Could you please specify your apartment preferences? For example:\n Desired neighborhoods or boroughs\n Number of bedrooms/bathrooms\n Budget range (monthly rent)\n Any amenities or must-have features\n Preferred move-in date\n This information will help me provide the most relevant apartment options in NYC."),
    HumanMessage(content="I'd prefer to live in Chelsea, Flatiron, or West Village. I'm looking for a 2 bed 2 bath, and I am looking for monthly rent below 7k. I'd like this to be a doorman building and have an in unit washer and dryer, but it's okay if there's no washer dryer. It's a plus if the building has a gym. And I'd like to move in in September 2025."),
]

conversation_3 = [
    HumanMessage(content="I need to research senior software developer salaries."),
    AIMessage(content="Could you specify the target location, industry, experience level, tech stack, and whether remote roles are included?"),
    HumanMessage(content="Focus on remote US-based roles in the FinTech industry for senior developers with 7+ years of experience specializing in Django and PostgreSQL.")
]

conversation_4 = [
    HumanMessage(content="I want to buy a new laptop for work."),
    AIMessage(content="To help you find the best laptop, could you specify your OS preference (Mac/Windows), budget, type of work you do (e.g., programming, design, office work), and any portability or battery life requirements?"),
    HumanMessage(content="I prefer a Mac. My budget is around $1500. I do mostly software development and need something portable with at least 12 hours of battery life.")
]

conversation_5 = [
    HumanMessage(content="We need to migrate our legacy transactional database system to the cloud, but we have very strict compliance and architectural constraints."),
    AIMessage(content="To help scope your research, could you specify:\n1. Your current database engine and data volume?\n2. Target cloud providers or any restrictions?\n3. Regulatory compliance requirements?\n4. Downtime tolerance for cutover?\n5. Budget and integration requirements?"),
    HumanMessage(content="We are migrating 15TB of relational data currently on Oracle 19c on-premise. We strictly CANNOT use AWS due to parent company vendor restrictions; we must focus exclusively on GCP or Azure hybrid solutions. The system must achieve SOC2 Type II and HIPAA compliance, with a hard budget ceiling of $8,000/month. We require near-zero downtime (<15 minutes total cutover window). Also, it is critical that the target database has native CDC (Change Data Capture) connectors for Kafka without third-party licensing fees like GoldenGate. Do not suggest NoSQL alternatives as we require complex multi-table ACID transactions.")
]

criteria_1 = [
    "Current age is 25",
    "Desired retirement age is 45",
    "Current risk tolerance is high",
    "Interested in investing in stocks and ETFs",
    "Open to forms of investment beyond stocks and ETFs",
    "Investment account is a regular brokerage account",
]

criteria_2 = [
    "Looking for a 2 bed 2 bath apartment in Chelsea, Flatiron, or West Village",
    "Monthly rent below 7k",
    "Should be in a doorman building",
    "Ideally have an in unit washer and dryer but not strict",
    "Ideally have a gym but not strict",
    "Move in date is September 2025"
]

criteria_3 = [
    "Role is senior software developer",
    "Location/Work model is remote US-based",
    "Industry is FinTech",
    "Experience level is 7+ years",
    "Tech stack includes Django and PostgreSQL"
]

criteria_4 = [
    "OS preference is Mac",
    "Budget is around $1500",
    "Primary use case is software development",
    "Needs to be portable",
    "Battery life requirement is at least 12 hours"
]

criteria_5 = [
    "Current database is on-premise Oracle 19c with 15TB of data",
    "Target environment is strictly limited to GCP or Azure hybrid solutions",
    "Explicitly exclude AWS from recommendations",
    "Must be compliant with SOC2 Type II",
    "Must be compliant with HIPAA",
    "Monthly infrastructure budget ceiling is $8,000",
    "Maximum cutover downtime allowed is under 15 minutes",
    "Must support native Kafka Change Data Capture (CDC) without third-party licensing fees",
    "Exclude NoSQL databases and maintain multi-table ACID transaction support"
]


# Initialize the LangSmith client
langsmith_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# Create the dataset
dataset_name = "deep_research_scoping"

# Get or create the dataset safely without deleting previous experiments
if not langsmith_client.has_dataset(dataset_name=dataset_name):
    print(f"Creating dataset '{dataset_name}'...")
    dataset = langsmith_client.create_dataset(
        dataset_name=dataset_name,
        description="A dataset that measures the quality of research briefs generated from an input conversation",
    )
else:
    print(f"Dataset '{dataset_name}' already exists. Fetching existing dataset...")
    dataset = langsmith_client.read_dataset(dataset_name=dataset_name)

# Add the examples to the dataset
langsmith_client.create_examples(
    dataset_id=dataset.id,
    examples=[
        {
            "inputs": {"messages": conversation_1},
            "outputs": {"criteria": criteria_1},
        },
        {
            "inputs": {"messages": conversation_2},
            "outputs": {"criteria": criteria_2},
        },
        {
            "inputs": {"messages": conversation_3},
            "outputs": {"criteria": criteria_3},
        },
        {
            "inputs": {"messages": conversation_4},
            "outputs": {"criteria": criteria_4},
        },
        {
            "inputs": {"messages": conversation_5},
            "outputs": {"criteria": criteria_5},
        },
    ],
)
print(f"Successfully uploaded 5 examples to dataset '{dataset_name}'.")