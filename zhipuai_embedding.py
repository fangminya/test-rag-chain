# LangChain封装ZhipuAI Embeddings
# 导入必要的库
import os
from typing import List, Optional
from langchain_core.embeddings import Embeddings

# 定义一个ZhipuAI Embeddings类，继承自Embeddings类
class ZhipuAIEmbeddings(Embeddings):
    def __init__(self):
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))


    # 重写embed_documents方法，对字符串计算Embedding
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        all_embeddings = []
        batch_size = 64  # 智谱AI API限制，最多64条
        
        # 分批处理文档
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model="embedding-3",
                    input=batch_texts
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                print(f"已处理 {i + len(batch_texts)}/{len(texts)} 个文档")
            except Exception as e:
                print(f"处理批次 {i//batch_size + 1} 时出错: {e}")
                # 如果批量处理失败，尝试逐个处理
                for text in batch_texts:
                    try:
                        response = self.client.embeddings.create(
                            model="embedding-3",
                            input=[text]
                        )
                        all_embeddings.append(response.data[0].embedding)
                    except Exception as e2:
                        print(f"处理单个文档时出错: {e2}")
                        # 如果单个文档也失败，添加空向量
                        all_embeddings.append([0.0] * 1024)  # embedding-3的维度是1024
        
        return all_embeddings


    # 重写embed_query方法，对单个字符串计算Embedding    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """

        return self.embed_documents([text])[0]

