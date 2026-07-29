# 知识库（RAG 检索语料）

本目录用于存放 HLE Agent 的领域知识检索语料。放入资料后，运行时会自动被
`hle_agent/rag_retriever.py` 扫描、分块并建立 TF-IDF 索引，在每道题求解前作为
参考材料（Retrieved reference material）注入到 Solver 上下文中。

## 支持格式
- `.txt` / `.md` / `.json` 文本文件
- 支持子目录（递归扫描）
- PDF 等二进制格式暂不支持，请先转为文本

## 使用方式
1. 把资料（如密码学手册、SageMath 教程、Python 语义说明、课程笔记等）放入本目录
2. 直接运行 `python run.py`，Agent 会自动检索相关片段
3. 资料为空时，Agent 退化为纯模型推理（不影响管线）

## 进阶：向量检索
当前默认使用零依赖的 TF-IDF 检索（`SimpleRAGRetriever`）。
如需更高质量的语义检索，可在 `rag_retriever.py` 中提供 embedding 函数并切换到
`VectorRAGRetriever`（余弦相似度）。embed_fn 可为本地 sentence-transformers 或任意
embedding API：

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
retriever = VectorRAGRetriever(embed_fn=lambda texts: model.encode(texts).tolist())
```

## 建议资料方向（针对本 20 题）
- 古典密码学 / 替换密码频率分析
- SageMath / 有限域多项式插值（Shamir 共享）
- Python 2/3 语义差异、真值语义、集合/元组运算
- C 语言语义陷阱、运算符优先级
- 矩阵秩 / ReLU / 神经网络表达能力
- 信号处理：重叠相加 / 重叠保留法 DFT 运算量
- 异构计算机架构（Bagua / Wuxing）数值类型语义
- 强化学习熵最大化、POMDP 记忆下界
- Martin Fowler 领域模型设计原则
