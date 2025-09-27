class MathTools:
    """数学计算工具集成"""
    
    def __init__(self, wolfram_app_id: str = None):
        self.wolfram_app_id = wolfram_app_id
        self.wolfram_client = None if not wolfram_app_id else wolframalpha.Client(wolfram_app_id)
    
    @tool
    def solve_math_expression(self, expression: str) -> str:
        """
        使用SymPy解决数学表达式
        支持代数、微积分、方程求解等
        """
        try:
            # 清理表达式
            cleaned_expr = self._clean_expression(expression)
            
            # 尝试不同类型的数学计算
            result = self._try_different_solvers(cleaned_expr)
            
            if result:
                return f"计算结果: {result}"
            else:
                return f"无法解析表达式: {expression}"
                
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    @tool
    def wolfram_alpha_query(self, query: str) -> str:
        """
        使用Wolfram Alpha进行数学查询
        支持更复杂的数学问题和自然语言查询
        """
        if not self.wolfram_client:
            return "Wolfram Alpha未配置"
        
        try:
            res = self.wolfram_client.query(query)
            results = []
            
            for pod in res.pods:
                if pod.title and pod.text:
                    results.append(f"{pod.title}: {pod.text}")
            
            if results:
                return "\n".join(results[:3])  # 返回前3个结果
            else:
                return "未找到相关结果"
                
        except Exception as e:
            return f"Wolfram Alpha查询错误: {str(e)}"
    
    @tool
    def math_formula_search(self, formula_query: str) -> str:
        """
        使用RAG系统搜索数学公式
        针对冷门数学公式查询
        """
        global rag_system  # 使用前面创建的RAG系统
        
        try:
            results = rag_system.retrieve(formula_query, k=3)
            
            if not results:
                return "未找到相关数学公式"
            
            response = "找到以下相关数学公式:\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['content']}\n"
                if 'metadata' in result and 'formula_count' in result['metadata']:
                    response += f"   (包含{result['metadata']['formula_count']}个公式)\n"
            
            return response
            
        except Exception as e:
            return f"公式搜索错误: {str(e)}"
    
    def _clean_expression(self, expr: str) -> str:
        """清理数学表达式"""
        # 移除不必要的空格和符号
        expr = expr.strip()
        expr = re.sub(r'\s+', ' ', expr)
        
        # 转换常见数学符号
        replacements = {
            '×': '*',
            '÷': '/',
            '·': '*',
            '∗': '*',
            '／': '/',
            '＋': '+',
            '－': '-',
            '＝': '=',
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
            '√': 'sqrt',
            'π': 'pi',
            '∞': 'oo',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'θ': 'theta',
            'λ': 'lambda',
            'μ': 'mu',
            'σ': 'sigma',
            'ω': 'omega'
        }
        
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        
        return expr
    
    def _try_different_solvers(self, expr: str) -> Optional[str]:
        """尝试不同类型的数学求解器"""
        try:
            # 1. 尝试简化表达式
            simplified = sympy.simplify(expr)
            if str(simplified) != expr:
                return str(simplified)
            
            # 2. 尝试求值（如果是数值表达式）
            if self._is_numeric_expression(expr):
                evaluated = sympy.N(expr)
                return str(evaluated)
            
            # 3. 尝试解方程
            if '=' in expr:
                left, right = expr.split('=', 1)
                equation = sympy.Eq(sympy.sympify(left), sympy.sympify(right))
                solutions = sympy.solve(equation)
                if solutions:
                    return str(solutions)
            
            # 4. 尝试微分
            if 'd/d' in expr or 'diff' in expr:
                # 处理微分表达式
                return self._handle_derivative(expr)
            
            # 5. 尝试积分
            if '∫' in expr or 'int' in expr:
                # 处理积分表达式
                return self._handle_integral(expr)
            
            # 6. 默认返回简化结果
            return str(simplified)
            
        except Exception:
            return None
    
    def _is_numeric_expression(self, expr: str) -> bool:
        """判断是否为数值表达式"""
        try:
            # 移除变量
            numeric_expr = re.sub(r'[a-zA-Z_][a-zA-Z0-9_]*', '1', expr)
            # 尝试求值
            sympy.N(numeric_expr)
            return True
        except Exception:
            return False
    
    def _handle_derivative(self, expr: str) -> Optional[str]:
        """处理微分表达式"""
        try:
            # 简单的微分处理
            if 'd/d' in expr:
                parts = expr.split('d/d')
                if len(parts) >= 2:
                    var = parts[1].split()[0]
                    func = parts[0].strip()
                    derivative = sympy.diff(sympy.sympify(func), sympy.Symbol(var))
                    return str(derivative)
            return None
        except Exception:
            return None
    
    def _handle_integral(self, expr: str) -> Optional[str]:
        """处理积分表达式"""
        try:
            # 简单的积分处理
            if '∫' in expr:
                # 移除积分符号
                clean_expr = expr.replace('∫', '').strip()
                integral = sympy.integrate(sympy.sympify(clean_expr))
                return str(integral)
            return None
        except Exception:
            return None

# 初始化数学工具
math_tools = MathTools(wolfram_app_id="YOUR_WOLFRAM_APP_ID")  # 替换为实际的Wolfram App ID

# 创建工具列表
tools = [
    Tool(
        name="SolveMathExpression",
        func=math_tools.solve_math_expression,
        description="使用SymPy解决数学表达式，支持代数、微积分、方程求解等"
    ),
    Tool(
        name="WolframAlphaQuery",
        func=math_tools.wolfram_alpha_query,
        description="使用Wolfram Alpha进行数学查询，支持更复杂的数学问题和自然语言查询"
    ),
    Tool(
        name="MathFormulaSearch",
        func=math_tools.math_formula_search,
        description="使用RAG系统搜索数学公式，针对冷门数学公式查询"
    )
]