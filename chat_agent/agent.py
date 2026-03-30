from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
import requests
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


load_dotenv()

checkpointer = InMemorySaver()

API_URL = os.environ['API_URL']

SYSTEM_PROMPT =  "You are a helpful assistant that provides sales forecasts based on user queries. " \
    "Do NOT call the get_sales_forecast tool unless you have at least one valid, specific parameter value from the user. If the user has not provided any of date, country, category, or device, ask them for details before calling the tool. " \
    "You can use the get_sales_forecast tool to retrieve predictions for specific dates, countries, product categories, and device types. " \
    "When a user asks for a sales forecast, determine the relevant parameters (date, country, category, device) from their query and call the get_sales_forecast tool with those parameters. " \
    "At least one of the 4 parameters must be present. If the user does not specify a parameter, you can call the tool with None for that parameter. " \
    "We may be able to infer the date from the context (e.g. if the user says 'next week', you can calculate the date 7 days from the current date). " \
    "Provide the predicted sales amount in your response to the user." \
    "You are an internal tool, so you do not need to explain your reasoning to the user. Just provide the forecast based on the parameters you can extract from the user's query." \
    "If the user is asking none sales forecast related questions, you can respond normally without calling the tool. Always steer the conversation towards sales forecast related queries when possible." \
    "Your job is not to provide general information about sales forecasting, but to provide specific forecasts based on the user's query. " \
    "If the user asks for a forecast but does not provide enough information, you can ask them for more details (e.g. 'Could you please specify the date or country you're interested in?') to help you call the tool with the right parameters." \
    "ALWAYS remind the user that predictions are more accurate when more specific parameters are provided. For example, if the user asks for a forecast without specifying a country, you can say 'I can provide a more accurate forecast if you specify the country you're interested in. Would you like to provide that information?'" \
    f"\nToday's date is {datetime.now().strftime('%Y-%m-%d')}."

@tool
def get_sales_forecast(date: str = None, country: str = None, category: str = None, device: str = None) -> float:
    """Call the sales forecast API to get a prediction.
    
    country must be one of: Sweden, Finland, Portugal, Spain, UK, France, Netherlands, Belgium, Bulgaria, Luxembourg, Italy, Ireland, Germany, Denmark, Austria
    category must be one of: Books, Games, Clothing, Beauty, Accessories, Appliances, Smartphones, Outdoors, Electronics, Other
    device must be one of: Mobile, PC, Tablet
    date format: YYYY-MM-DD
    Only pass parameters the user has explicitly specified.
    Remember that predictions are more accurate when more specific parameters are provided, so encourage the user to provide as many details as possible for the best forecast. If the user has not provided any parameters, ask them for more details before calling this tool.
    When you give the predicted number back to the user, please include a euro sign and format it with commas for thousands (e.g. €1,234.56).
    """
    date = None if not date or date.lower() in ("null", "none") else date
    country = None if not country or country.lower() in ("null", "none") else country
    category = None if not category or category.lower() in ("null", "none") else category
    device = None if not device or device.lower() in ("null", "none") else device

    if not any([date, country, category, device]):
        raise ValueError("At least one parameter must be provided")

    payload = {}
    if date: payload["date"] = date
    if country: payload["country"] = country
    if category: payload["category"] = category
    if device: payload["device_type"] = device

    response = requests.post(f"{API_URL}/predict", json=payload)
    if response.status_code == 200:
        return response.json()['predicted_sales']
    else:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    

model = init_chat_model(
    "claude-sonnet-4-5",
    temperature=0
)
    
agent = create_agent(
    model=model,
    tools=[get_sales_forecast],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

def send_agent_query(query: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    message = {"messages": [{"role": "user", "content": query}]}
    response = agent.invoke(message, config=config)
    return response['messages'][-1].content


# --- FastAPI server ---

app = FastAPI(title="Sales Forecast Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    response = send_agent_query(request.message, thread_id=request.thread_id)
    return ChatResponse(response=response)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)