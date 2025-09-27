class MathFormulaRAG:
    """数学公式RAG系统"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = MathFormulaEmbeddings()
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
    
    def load_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """加载文档到向量数据库"""
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # 使用数学公式专用文本分割器
        text_splitter = MathFormulaTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = []
        split_metadatas = []
        
        for i, doc in enumerate(documents):
            doc_splits = text_splitter.split_text(doc)
            splits.extend(doc_splits)
            # 为每个分割块添加元数据
            for j, split in enumerate(doc_splits):
                meta = metadatas[i].copy()
                meta.update({
                    "chunk_id": f"doc_{i}_chunk_{j}",
                    "original_doc_id": i,
                    "chunk_index": j,
                    "formula_count": len(re.findall(r'\$.*?\$|\$\$.*?\$\$', split))
                })
                split_metadatas.append(meta)
        
        # 创建或加载ChromaDB
        self.vectorstore = Chroma.from_texts(
            texts=splits,
            embedding=self.embeddings.embeddings,
            metadatas=split_metadatas,
            persist_directory=self.persist_directory
        )
        
        # 创建检索器
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # 返回前3个最相关的结果
        )
        
        # 创建QA链
        from langchain.chains import RetrievalQA
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=sky_t1_llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """检索相关文档"""
        # 增强查询的数学公式表示
        enhanced_query = self.embeddings.enhance_math_embedding(query)
        
        # 执行检索
        docs = self.retriever.get_relevant_documents(enhanced_query)
        
        # 格式化结果
        results = []
        for i, doc in enumerate(docs[:k]):
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, "score", 1.0 - i*0.1),  # 模拟相关度分数
                "rank": i + 1
            })
        
        return results
    
    def query(self, question: str) -> Dict:
        """问答查询"""
        # 增强查询
        enhanced_question = self.embeddings.enhance_math_embedding(question)
        
        # 执行QA
        result = self.qa_chain({"query": enhanced_question})
        
        return {
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result["source_documents"]
            ]
        }

# 使用示例
rag_system = MathFormulaRAG()

# 假设有数学公式文档
math_documents = [
    "勾股定理：在直角三角形中，斜边的平方等于两直角边的平方和，即 $c^2 = a^2 + b^2$。",
    "二次方程求根公式：对于方程 $ax^2 + bx + c = 0$，其解为 $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$。",
    "欧拉公式：$e^{i\\pi} + 1 = 0$，这是数学中最优美的公式之一。",
    "微积分基本定理：如果 $F(x)$ 是 $f(x)$ 的一个原函数，则 $\\int_a^b f(x) dx = F(b) - F(a)$。",
    "泰勒展开：$f(x) = f(a) + f'(a)(x-a) + \\frac{f''(a)}{2!}(x-a)^2 + \\cdots + \\frac{f^{(n)}(a)}{n!}(x-a)^n + R_n(x)$。"
]

# 加载文档
rag_system.load_documents(math_documents)