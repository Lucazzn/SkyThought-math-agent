import os



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


