from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
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

    researcher = AssistantAgent(
        name="REsearcher",
        model_client=model_client,
        
        system_message="""Research the topic thoroughly and provide facts.
    Do not write the final blog."""
    )

    writter = AssistantAgent(
        name="Writter",
        model_client=model_client,
        system_message="Use the researcher's findings to write a concise blog.."
    )

    critic = AssistantAgent(
        name="Critic",
        model_client=model_client,
        system_message="""Review the writer's blog.
    Suggest improvements.
    If no improvements are needed,
    respond only with FINISH."""
    )

    termination1 = TextMentionTermination("FINISH")
    termination2 =  MaxMessageTermination(max_messages=6)

  



    team = SelectorGroupChat(
        allow_repeated_speaker=True,
         model_client=model_client, 
        participants=[critic,writter,researcher],
        termination_condition=termination1 | termination2
    )
    
    await Console(
        team.run_stream(
            task="""Research science.
Writer should write a blog based on the research.
Critic should review the blog."""
        )
    )

asyncio.run(main())