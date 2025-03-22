from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse) # This is the output parser that will be used to parse the response from the model
prompt = ChatPromptTemplate(
    [
        ("system",
         """You are a research assistant that will help me work on a project. "
         Answer the user query and use neccessary tools to research the topic. "
         Wrap the output in this format and provide no other text\n{format_instructions}"""),
         ("placeholder", "{chat_history}"),
         ("human", "{query}"),
         ("placeholder", "{agent_scratchpad}"),
    ],
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools=[]
)

agent_executor = AgentExecutor(agent = agent, tools = [], verbose = True)
raw_response = agent_executor.invoke(({ "query": "What is the capital of France?" }))
print(raw_response)


