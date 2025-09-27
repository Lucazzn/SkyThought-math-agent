# # 安装必要依赖
# !pip install langchain
# !pip install langchain-community
# !pip install chromadb
# !pip install sympy
# !pip install wolframalpha
# !pip install sentence-transformers
# !pip install torch transformers accelerate

# # 如果使用vLLM部署
# !pip install vllm

# 导入必要库
import os
import re
import json
import sympy
import wolframalpha
from typing import List, Dict, Any, Optional, Tuple
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool, Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import VLLM


from vllm import LLM, SamplingParams

def react_cycle(query):
    # 思考阶段：分析问题意图
    intent = classify_intent(query)  # 分类器判断问题类型
    
    # 行动阶段：选择并执行工具
    if intent == "calculation":
        tool = select_tool("SymPy")  # 选择计算工具
        result = execute_tool(tool, query)
    elif intent == "formula_lookup":
        tool = select_tool("RAG")  # 选择检索工具
        result = execute_tool(tool, query)
    else:
        tool = select_tool("Wolfram Alpha")
        result = execute_tool(tool, query)
    
    # 观察阶段：验证结果并调整
    if not validate_result(result):
        # 如果结果不合理，调整策略重新执行
        adjust_strategy()
        return react_cycle(query)
    
    return result

# 初始化vLLM模型
llm = LLM(
    model="Sky-T1-7B",  # 替换为实际模型路径
    tensor_parallel_size=1,  # 根据GPU数量调整
    max_model_len=4096,
    dtype="float16"
)

# 创建LangChain兼容的LLM包装器
from langchain_community.llms import VLLM as LangChainVLLM

sky_t1_llm = LangChainVLLM(
    model="Sky-T1-7B",
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    tensor_parallel_size=1
)

# 方式2：直接使用Transformers（如果vLLM不可用）
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Sky-T1-7B")
model = AutoModelForCausalLM.from_pretrained(
    "Sky-T1-7B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 创建LangChain兼容的LLM
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95
)

sky_t1_llm = HuggingFacePipeline(pipeline=pipe)



