import sys
import os
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv, find_dotenv

# 将父目录放入系统路径中
sys.path.append('/chroma')

#---------------------------------------------- 获取 zhipu 的 api_key 
 # 1、加载 .env 文件,读取 api_key
load_dotenv('zhipuapi.env')
# 获取 api_key
zhipuai_api_key = os.getenv('ZHIPUAI_API_KEY')

#--------------------------------------------- 获取 zhipu 的 Embedding模型 
# 定义 Embeddings
embedding = ZhipuAIEmbeddings()

#--------------------------------------------- 加载本地向量数据库 Chroma 
# 向量数据库持久化路径
persist_directory = '/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)

#--------------------------------------------- 构建检索链
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 我们可以使用LangChain的LCEL(LangChain Expression Language, LangChain表达式语言)来构建workflow
from langchain_core.runnables import RunnableLambda
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

combiner = RunnableLambda(combine_docs)
retrieval_chain = retriever | combiner

#--------------------------------------------- 构建 zhipu 大模型 llm
from zhipuai_llm import ZhipuaiLLM

llm = ZhipuaiLLM(model_name="glm-4-plus", temperature=0.1, api_key=zhipuai_api_key)


#--------------------------------------------- 构建检索问答链
# 构建检索问答链
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。请你在回答的最后说“谢谢你的提问！”。
{context}
问题: {input}
"""
# 将template通过 PromptTemplate 转为可以在LCEL中使用的类型
prompt = PromptTemplate(template=template, input_variables=["context", "input"])

qa_chain = (
    RunnableParallel({"context": retrieval_chain, "input": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

# question_1 = "什么是南瓜书？"
# question_2 = "Prompt Engineering for Developer是谁写的？"

# result = qa_chain.invoke(question_1)
# print("大模型+知识库后回答 question_1 的结果：")
# print(result)

# result = qa_chain.invoke(question_2)
# print("大模型+知识库后回答 question_2 的结果：")
# print(result)

#--------------------------------------------- 向检索链添加聊天记录
from langchain_core.prompts import ChatPromptTemplate

# 问答链的系统prompt
system_prompt = (
    "你是一个问答任务的助手。 "
    "请使用检索到的上下文片段回答这个问题。 "
    "如果你不知道答案就说不知道。 "
    "请使用简洁的话语回答用户。"
    "\n\n"
    "{context}"
)
# 制定prompt template
qa_prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

#--------------------------------------------- 带有信息压缩的检索链
from langchain_core.runnables import RunnableBranch

# 压缩问题的系统 prompt
condense_question_system_template = (
    "请根据聊天记录完善用户最新的问题，"
    "如果用户最新的问题不需要完善则返回用户的问题。"
    )
# 构造 压缩问题的 prompt template
condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
# 构造检索文档的链
# 简化版本：直接使用用户输入进行检索
retrieve_docs = (lambda x: x["input"]) | retriever

# 重新定义 combine_docs_with_context
def combine_docs_with_context(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"]) # 将 docs 改为 docs["context"]
# 定义问答链
qa_chain = (
    RunnablePassthrough.assign(context=combine_docs_with_context) # 使用 combine_docs_with_context 函数整合 qa_prompt 中的 context
    | qa_prompt # 问答模板
    | llm
    | StrOutputParser() # 规定输出的格式为 str
)
# 定义带有历史记录的问答链
qa_history_chain = RunnablePassthrough.assign(
    context = (lambda x: x) | retrieve_docs # 将查询结果存为 content
    ).assign(answer=qa_chain) # 将最终结果存为 answer

# 不带聊天记录
input_data = {
    "input": "西瓜书是什么？",
    "chat_history": []
}
result = qa_history_chain.invoke(input_data)
print("=== 不带聊天记录 ===")
print(f"输入问题: {input_data['input']}")
print(f"聊天历史: {input_data['chat_history']}")
print(f"检索到的上下文: {result['context']}")
print(f"最终回答: {result['answer']}")
print("=" * 50)

# 带聊天记录
input_data = {
    "input": "南瓜书跟它有什么关系？",
    "chat_history": [
        ("human", "西瓜书是什么？"),
        ("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
    ]
}
result = qa_history_chain.invoke(input_data)
print("=== 带聊天记录 ===")
print(f"输入问题: {input_data['input']}")
print(f"聊天历史: {input_data['chat_history']}")
print(f"检索到的上下文: {result['context']}")
print(f"最终回答: {result['answer']}")
print("=" * 50)