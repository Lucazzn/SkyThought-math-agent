import os

from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from typing import List, Union, Any
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import Tool

from MyAgentTool import MyAgentTool

# 使用Tavily 搜索引擎:https://app.tavily.com/home
os.environ["TAVILY_API_KEY"]="TAVILY_API_KEY"
os.environ["LANGCHAIN_API_KEY"]="LANGCHAIN_API_KEY"
os.environ["DASHSCOPE_API_KEY"]="DASHSCOPE_API_KEY"
class MyAgent:
    def __init__(self)->None:
# Agent 的提示词模板
        self.template ="""请尽可能详细地回答下面的问题，你将始终用中文回答。当需要时，你可以使用以下工具:
                        {tools}
                        请按照以下格式回答:
                        Question: {input}
                        Thought: 你应该思考下一步该做什么
                        Action: 选择一个操作，必须是以下工具之一 [{tool_names}]
                        Action Input: 该操作的输入
                        Observation: 该操作的结果
                        ...（此 Thought/Action/Action Input/Observation 可以重复多次）
                        Thought: 我现在知道最终的答案了
                        Final Answer: {final_answer}
                        现在开始! 记住使用中文回答，如果使用英文回答将受到惩罚。
                        Question: {input}
                        {agent_scratchpad}"""

# 定义语言模型（LLM）
        self.llm =ChatTongyi(model='qwen-plus')

# 初始化工具列表
        self.tools =MyAgentTool().tools()

# 创建 Agent 的提示词
        self.prompt = self.MyTemplate(
            template=self.template,
            tools=self.tools,
            input_variables=["input","intermediate_steps"],
)

# 定义 LLMChain
        self.llm_chain =LLMChain(
            llm=self.llm,
            prompt=self.prompt
)

# 获取工具名称列表
        self.toolnames =[tool.name for tool in self.tools]

# 定义一个 LLMSingleActionAgent
        self.agent =LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            allowed_tools=self.toolnames,
            output_parser=self.MyOutputParser(),
            stop=["\nObservation:"],
)

# 运行 Agent 的方法
    def run(self,input: str)->str:
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=True
        )
        return agent_executor.run(input=input)

# 自定义模板渲染类
class MyTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self,**kwargs: Any)->str:
# 获取中间步骤
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts =""
        
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

# 将 agent_scratchpad 设置为该值
        kwargs["agent_scratchpad"]= thoughts

# 从提供的工具列表中创建一个名为 tools 的变量
        kwargs["tools"]="\n".join([f"{tool.name}: {tool.description}"for tool in self.tools])

# 创建一个提供的工具名称列表
        kwargs["tool_names"]=", ".join([tool.name for tool in self.tools])

        # 确保传递 final_answer 变量
        if"final_answer" not in kwargs:
                    kwargs["final_answer"]="暂时没有答案"

        return self.template.format(**kwargs)

# 自定义输出解析类
class MyOutputParser(AgentOutputParser):
    def parse(self,output: str)-> Union[AgentAction, AgentFinish]:
        if"Final Answer:"in output:
            
            return AgentFinish(
                return_values={"output": output.split("Final Answer:")[-1].strip()},
                log=output,
            )
# 使用正则解析出动作和动作输入
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\s*:(.*)"
        match = re.search(regex, output, re.DOTALL)
        if not match:
            raise OutputParserException(f"无法解析 LLM 输出: `{output}`")
        action = match.group(1).strip()
        action_input = match.group(2).strip(" ").strip('"')

# 返回操作和操作输入
        
        return AgentAction(tool=action, tool_input=action_input, log=output)

if __name__ =="__main__":
    myagent =MyAgent()
    question ="请计算以下表达式的结果：2 + 3 * 4 - 5 / 2"
    result = myagent.run(question)
    print("Agent 的回答:", result)
    question ="请比较以下两个数字的大小：10, 20"
    result = myagent.run(question)
    print("Agent 的回答:", result)
    
    
#   easy tool test    
    from langchain_community.tools.tavily_search import TavilySearchResults

    # 使用Tavily 搜索引擎:https://app.tavily.com/home
    os.environ["TAVILY_API_KEY"]="TAVILY_API_KEY"
    os.environ["LANGCHAIN_API_KEY"]="LANGCHAIN_API_KEY"

    search =TavilySearchResults(max_results=2)
    search_results = search.invoke("深圳天气怎么样？")
    print(search_results)
    # If we want, we can create other tools.
    # Once we have all the tools we want, we can put them in a list that we will reference later.
    tools =[search]
 