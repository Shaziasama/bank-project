import os
from dotenv import load_dotenv
from agents import Agent,Runner,OpenAIChatCompletionsModel,AsyncOpenAI,RunConfig,function_tool,enable_verbose_stdout_logging
import random

from pydantic import BaseModel, Field
enable_verbose_stdout_logging()



import rich
load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

class service_type(BaseModel):
   service: str
   confidence: float
   keywords_dectected: list[str]
   reasoning: str




class ToolInfo(BaseModel):
    token_number: str 
    wait_time: str 
    message: str 
    service_type: str

@function_tool
def identify_baking_purpose(customer_request: str) -> str:
    """It is a simple function to figure out what banking service customer needs."""
    request = customer_request.lower()
    if ("balance" in request) or ("account" in request) or ("statement" in request):
       return service_type(
           service="account_services",
           confidence=0.9,
           keywords_dectected=["balance","account","statement"],
           reasoning="Customer wants to check their account"
       )
    elif ("transfer" in request) or ("send" in request) or ("payment" in request):
       return service_type(
           service="transfer_services",
           confidence=0.9,
           keywords_dectected=["transfer", "send","payment"],
           reasoning="Customer wants to send money to make payment"
       )
    elif ("loan" in request) or ("mortgage" in request) or ("borrow" in request):
       return service_type(
           service="loan_services",
           confidence=0.9,
           keywords_dectected=["loan", "mortgage","borrow"],
           reasoning="Customer needs help with a loan or mortgage"
       )
    else:
       return service_type(
           service="general_banking",
           confidence=0.5,
           keywords_dectected=["general","help","query"],
           reasoning="Customer has a general_banking help"
       )




@function_tool
def generate_customer_token(service_type: str = "general") -> ToolInfo:
    """Generate a token number for the customer query
    args:
    service_type = general
    service_type = account services
    service_type = transfer_services
    service_type = loan_services
    
"""
    if service_type == "account_services":
      prefix = "A"
      wait_time = "5-10 minutes"

    elif service_type == "transfer_services":
      prefix = "T"
      wait_time = "2.5 minutes"

    elif service_type == "loan_services":
      prefix = "L"
      wait_time = "15-20 minutes"
    else:
      prefix = "G"
      wait_time = "8-10 minutes"
    token_number = f"{prefix}{random.randint(100, 999)}"

    return ToolInfo(
        token_number=token_number,
        wait_time=wait_time,
        message=f"please take token {token_number}. and wait for a {wait_time} and have a seat, we will call you shortly.",
        service_type=service_type
    )



account_agent = Agent(
    name="Account Services Agent",
    instructions="you help user in their quary of account balance, statements and account information always generate a token"
)

transfer_agent = Agent(
    name="Transfer Services Agent",
    instructions="you help user with money transfer and payments, always generate a token"
)

loan_agent = Agent(
    name="Loan Services Agent",
    instructions="you help user with loans and mortgagges always generate token"
)



agent = Agent(
    name="Bank Greeting Agent",
    instructions="""you are friendly bank greeting agent.
    1. welcome customer nicely.
    2. user identify_baking_purpose to understand user need.
    3. if confidence > 0.8, send user to the right specialist.
    4. otherwise generate a general token.
    5. always use generate_customer_token tool to generate token.


    #example: argument for the generate_customer_token can only be (service_type = "general" or service_type "account_services" or service_type = "transfer_services" or service_type = "loan_services")
    
    Always be helpful
    """,
    handoffs=[account_agent, transfer_agent, loan_agent],
    tools=[generate_customer_token,identify_baking_purpose]
    
)

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",

)

config = RunConfig(
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash",openai_client=client),
    model_provider=client,
    tracing_disabled=True

)

while True:
    user_input = input("Enter your query: ")
    if user_input.lower() in ["exit", "quit"]:break
       

    result=Runner.run_sync(agent, input=user_input, run_config=config)
    rich.print(result.final_output) 





