class IntentClassifier:
    """意图分类器，判断用户问题类型"""
    
    def __init__(self, llm):
        self.llm = llm
        self.intent_categories = {
            "calculation": "数学计算（如求值、解方程、微积分等）",
            "formula_search": "数学公式查询（如询问特定公式、定理等）",
            "general_reasoning": "通用推理（如数学概念解释、证明思路等）",
            "unknown": "未知类型"
        }
    
    def classify_intent(self, query: str) -> str:
        """分类用户意图"""
        # 使用LLM进行意图分类
        prompt = f"""
        请对以下数学问题进行意图分类，选择最合适的类别：
        
        问题：{query}
        
        可选类别：
        1. calculation - 数学计算（如求值、解方程、微积分等）
        2. formula_search - 数学公式查询（如询问特定公式、定理等）
        3. general_reasoning - 通用推理（如数学概念解释、证明思路等）
        4. unknown - 未知类型
        
        请只返回类别名称，不要添加其他内容。
        """
        
        try:
            # 使用LLM进行分类
            response = self.llm.invoke(prompt)
            intent = response.strip().lower()
            
            # 验证返回的意图是否在预定义类别中
            for category in self.intent_categories.keys():
                if category in intent:
                    return category
            
            # 如果LLM返回不在预定义类别中，使用规则进行判断
            return self._rule_based_classification(query)
            
        except Exception as e:
            print(f"意图分类错误: {e}")
            return self._rule_based_classification(query)
    
    def _rule_based_classification(self, query: str) -> str:
        """基于规则的意图分类（备用方案）"""
        query_lower = query.lower()
        
        # 计算相关关键词
        calc_keywords = ['计算', '求值', '等于多少', '解方程', '积分', '微分', '导数',