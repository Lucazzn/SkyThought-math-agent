import os



from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_community.llms import Tongyi
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 使用Tavily 搜索引擎:https://app.tavily.com/home
os.environ["TAVILY_API_KEY"]="TAVILY_API_KEY"
os.environ["LANGCHAIN_API_KEY"]="LANGCHAIN_API_KEY"
os.environ["DASHSCOPE_API_KEY"]="DASHSCOPE_API_KEY"

llm=Tongyi(temperature=1)
template='''
        你的名字是小黑子,当人问问题的时候,你都会在开头加上'唱,跳,rap,篮球!',然后再回答{question}
    '''
prompt=PromptTemplate(
        template=template,
        input_variables=["question"]#这个question就是用户输入的内容,这行代码不可缺少
)
chain =LLMChain(#将llm与prompt联系起来
        llm=llm,
        prompt=prompt
)
question='你是谁'
res=chain.invoke(question)#运行
print(res['text'])#打印结果


