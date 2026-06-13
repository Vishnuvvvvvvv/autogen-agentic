import asyncio


from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import Image
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_ext.tools.mcp import McpWorkbench
from autogen_ext.tools.mcp import StdioServerParams


async def main():

    model_client = OpenAIChatCompletionClient(
    model="gemini-3.1-flash-lite",
     model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
    api_key="",
    )


    server_params = StdioServerParams( 
        command= "npx",
      args= [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        r"F:\Other\agentic-ai-autogen-udemy"
       
      ],
      timeout=30)
    

    fs_workbench = McpWorkbench(server_params)

    async with fs_workbench as fs_wb:


      file_sm_assistant  = AssistantAgent(name="assistant",model_client=model_client,workbench=fs_wb,
                                  system_message="""You have access to filesystem tools.

Use tools when necessary.

When the task is finished,
respond with exactly:

TASK COMPLETE"""
      )

      human= UserProxyAgent(name="human")

      termination1 = TextMentionTermination("TASK COMPLETE")
      termination2 = MaxMessageTermination(max_messages=10)
      
      team = RoundRobinGroupChat(participants=[human,file_sm_assistant],
                                 termination_condition=termination1 | termination2)


      await Console(team.run_stream(task="Find all Python files under the workspace"))
      await model_client.close()


asyncio.run(main())
