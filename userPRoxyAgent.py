from typing import Text

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo


import asyncio

async def main():

    model_client = OpenAIChatCompletionClient(
        model="gemini-3.1-flash-lite",
         model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
        api_key=""
    )

    teacher = AssistantAgent(
        name="Teacher",
        model_client=model_client,
        
        system_message="You are a science teacher.ask follow up questions. When user says THANKS ,then say LESSON COMPLETE."
    )

    user_proxy = UserProxyAgent(
        name="Student",
       
    )

    team = RoundRobinGroupChat(
        participants=[teacher, user_proxy],
        termination_condition=TextMentionTermination("LESSON COMPLETE")
    )

    await Console(
        team.run_stream(
            task="Let us discuss about science.",
           
        )
    )

asyncio.run(main())