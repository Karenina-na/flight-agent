from langchain_openai import ChatOpenAI          # 使用这个替换
from langchain.agents import create_agent
from src.config import load_settings
from src.runtime import Context
from src.memory import checkpointer
from src.prompt import build_system_prompt
from src.tools import get_tools

settings = load_settings()

model = ChatOpenAI(
    base_url=settings.llm.base_url,
    api_key=settings.llm.api_key,
    model=settings.llm.model,
    temperature=settings.llm.temperature,
)

# 如果您的服务完全兼容 OpenAI，也可以继续使用 init_chat_model，但要传入 openai 前缀：
# model = init_chat_model(
#     "openai:google/gemma-4-e2b", 
#     temperature=0,
#     model_kwargs={"base_url": "http://127.0.0.1:1234/v1", "api_key": "not-needed"}
# )

agent = create_agent(
    model=model,
    system_prompt=build_system_prompt(tools=get_tools()),
    tools=get_tools(),
    context_schema=Context,
    checkpointer=checkpointer
)
