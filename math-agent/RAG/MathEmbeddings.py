class MathFormulaEmbeddings:
    """针对数学公式的嵌入模型"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        return self.embeddings.embed_query(text)
    
    def enhance_math_embedding(self, text: str) -> str:
        """增强数学公式的嵌入效果"""
        # 对数学公式进行预处理，提高嵌入质量
        enhanced_text = text
        
        # 将LaTeX公式转换为更易理解的文本描述
        # 例如：将 $\frac{a}{b}$ 转换为 "a divided by b"
        enhanced_text = self._convert_latex_to_text(enhanced_text)
        
        return enhanced_text
    
    def _convert_latex_to_text(self, text: str) -> str:
        """将LaTeX公式转换为文本描述"""
        # 基础转换规则
        conversions = [
            (r'\frac\{([^}]+)\}\{([^}]+)\}', r'\1 divided by \2'),
            (r'\sqrt\{([^}]+)\}', r'square root of \1'),
            (r'\int', 'integral'),
            (r'\sum', 'summation'),
            (r'\prod', 'product'),
            (r'\lim', 'limit'),
            (r'\to', 'to'),
            (r'\infty', 'infinity'),
            (r'\alpha', 'alpha'),
            (r'\beta', 'beta'),
            (r'\gamma', 'gamma'),
            (r'\delta', 'delta'),
            (r'\epsilon', 'epsilon'),
            (r'\pi', 'pi'),
            (r'\theta', 'theta'),
            (r'\lambda', 'lambda'),
            (r'\mu', 'mu'),
            (r'\sigma', 'sigma'),
            (r'\omega', 'omega'),
            (r'\pm', 'plus or minus'),
            (r'\times', 'times'),
            (r'\div', 'divided by'),
            (r'\cdot', 'dot'),
            (r'\neq', 'not equal to'),
            (r'\leq', 'less than or equal to'),
            (r'\geq', 'greater than or equal to'),
            (r'\approx', 'approximately equal to'),
            (r'\equiv', 'equivalent to'),
            (r'\subset', 'subset of'),
            (r'\supset', 'superset of'),
            (r'\in', 'in'),
            (r'\notin', 'not in'),
            (r'\forall', 'for all'),
            (r'\exists', 'there exists'),
            (r'\nabla', 'nabla'),
            (r'\partial', 'partial'),
            (r'\prime', 'prime'),
            (r'\degree', 'degrees'),
        ]
        
        for pattern, replacement in conversions:
            enhanced_text = re.sub(pattern, replacement, text)
        
        return enhanced_text

# 初始化嵌入模型
embeddings = MathFormulaEmbeddings()