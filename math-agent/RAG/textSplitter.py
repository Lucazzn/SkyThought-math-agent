class MathFormulaTextSplitter:
    """针对数学公式的文本分割器"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 使用RecursiveCharacterTextSplitter作为基础
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
    
    def split_text(self, text: str) -> List[str]:
        """分割文本，特别处理数学公式"""
        # 首先按段落分割
        chunks = self.text_splitter.split_text(text)
        
        # 对包含数学公式的块进行特殊处理
        enhanced_chunks = []
        for chunk in chunks:
            # 检测是否包含数学公式（LaTeX格式）
            if self._contains_math_formula(chunk):
                # 对数学公式块进行更精细的分割
                formula_chunks = self._split_formula_chunk(chunk)
                enhanced_chunks.extend(formula_chunks)
            else:
                enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _contains_math_formula(self, text: str) -> bool:
        """检测文本是否包含数学公式"""
        # 检测LaTeX数学公式标记
        math_patterns = [
            r'\$.*?\$',           # 行内公式 $...$
            r'\$\$.*?\$\$',      # 行间公式 $$...$$
            r'\\[a-zA-Z]+',      # LaTeX命令 \command
            r'\{.*?\}',          # LaTeX分组 {...}
            r'\d+/\d+',          # 分数 1/2
            r'\d+\^\d+',         # 幂运算 2^3
            r'\d+_\d+',          # 下标 a_1
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _split_formula_chunk(self, chunk: str) -> List[str]:
        """对包含数学公式的块进行特殊分割"""
        # 按公式分割
        parts = re.split(r'(\$.*?\$|\$\$.*?\$\$)', chunk)
        result = []
        current_chunk = ""
        
        for part in parts:
            if not part.strip():
                continue
                
            # 如果是数学公式
            if re.match(r'\$.*?\$|\$\$.*?\$\$', part):
                # 如果当前块不为空，先保存
                if current_chunk.strip():
                    result.append(current_chunk.strip())
                    current_chunk = ""
                # 保存公式块
                result.append(part.strip())
            else:
                # 如果是普通文本
                if len(current_chunk) + len(part) > self.chunk_size:
                    if current_chunk.strip():
                        result.append(current_chunk.strip())
                    current_chunk = part
                else:
                    current_chunk += part
        
        if current_chunk.strip():
            result.append(current_chunk.strip())
        
        return result

# 使用示例
text_splitter = MathFormulaTextSplitter(chunk_size=500, chunk_overlap=50)