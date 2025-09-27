from langchain.agents import Tool
from typing import Any

class MyAgentTool:
    def __init__(self)->None:
# 初始化工具，可以在此添加更多本地工具
        pass

    def tools(self):
        
        return[
            Tool(
                name="math",
                description="用于执行基本的数学计算，比如加减乘除。",
                func=self.math_tool,
            ),
            Tool(
                name="compare_numbers",
                description="比较两个数的大小，格式：'num1,num2'。",
                func=self.compare_numbers,
            )
    ]

    def math_tool(self,input: str)->str:
        """
        简单的数学计算工具，解析输入的数学表达式并返回结果。
        """
        try:
        # 安全地计算数学表达式
        # 仅允许数字和基本运算符
            allowed_chars ="0123456789+-*/(). "
            if not all(char in allowed_chars for char in input):
                return"输入包含不允许的字符。"
            result =eval(input)
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    def compare_numbers(self,input: str)->str:
        """
        比较两个数的大小，输入格式为 'num1,num2'。
        返回比较结果：num1 > num2, num1 < num2 或 num1 = num2。
        """
        try:
# 解析输入字符串为两个数字
            num1_str, num2_str = input.split(',')
            num1 =float(num1_str.strip())
            num2 =float(num2_str.strip())

            # 比较两个数字的大小
            if num1 >num2:
                return f"{num1} > {num2}"
            elif num1 <num2:
                return f"{num1} < {num2}"
            else:
                return f"{num1} = {num2}"
        except ValueError:
            return"输入格式错误，请使用 'num1,num2' 形式。"


# 实例化工具类
tools =MyAgentTool()

# 调用 math 工具
print(tools.math_tool("3 + 5 * 2"))# 输出: 13
print(tools.math_tool("10 / 2 - 3"))# 输出: 2.0
print(tools.math_tool("import os"))# 输出: 输入包含不允许的字符。


