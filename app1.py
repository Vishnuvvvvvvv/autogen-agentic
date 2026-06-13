import asyncio


from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.ui import Console
from autogen_core import Image
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

async def main():
    print("Hello, World!")
    
    model_client = OpenAIChatCompletionClient(
    model="gemini-3.1-flash-lite",
     model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
    api_key="",
    )


    image = Image.from_file("F:\\Resume\\newpic.jpg")

    message = MultiModalMessage(
    content=[
        "Describe what you see in this image.",
        image
    ],
    source="user"
    )
    
    assistant  = AssistantAgent(name="assistant",model_client=model_client)

    await Console(assistant.run_stream(task=message))
    await model_client.close()


asyncio.run(main())
