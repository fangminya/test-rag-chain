import streamlit as st
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv
from zhipuai_embedding import ZhipuAIEmbeddings
from zhipuai_llm import ZhipuaiLLM
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

# 加载环境变量
load_dotenv()


# 定义get_retriever函数，该函数返回一个检索器
def get_retriever():
    embedding = ZhipuAIEmbeddings()
    chroma_api_key = os.getenv("CHROMA_API_KEY") or os.getenv("chroma_api_key")
    chroma_cloud_host = os.getenv("CHROMA_CLOUD_HOST") or os.getenv("chroma_cloud_host")
    st.write("chroma_api_key:", chroma_api_key, type(chroma_api_key))
    st.write("chroma_cloud_host:", chroma_cloud_host, type(chroma_cloud_host))
    chroma_api_key = str(chroma_api_key or "").strip()
    chroma_cloud_host = str(chroma_cloud_host or "").strip()
    if not chroma_api_key or not chroma_cloud_host:
        st.error("请在 Streamlit 的 Secrets 中配置 chroma_api_key 和 chroma_cloud_host")
        st.stop()
    settings = Settings(
        chroma_api_impl="rest",
        chroma_server_host=chroma_cloud_host,
        chroma_server_http_headers={"Authorization": f"Bearer {chroma_api_key}"}
    )
    st.write("Settings:", settings)
    vectordb = Chroma(
        embedding_function=embedding,
        client_settings=settings
    )
    return vectordb.as_retriever()

# 定义combine_docs函数， 该函数处理检索器返回的文本
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])


# 定义get_qa_history_chain函数，该函数可以返回一个检索问答链
def get_qa_history_chain():
    retriever = get_retriever()
    # 从环境变量获取 API key
    api_key = os.getenv('ZHIPUAI_API_KEY')
    if not api_key:
        st.error("请在 Streamlit 的 Secrets 中配置 ZHIPUAI_API_KEY")
        return None
    llm = ZhipuaiLLM(model_name="glm-4-plus", temperature=0, api_key=api_key)
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    # 简化版本：直接使用用户输入进行检索
    retrieve_docs = (lambda x: x["input"]) | retriever

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain


# 定义gen_response函数，它接受检索问答链、用户输入及聊天历史，并以流式返回该链输出
def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]


# 定义main函数，该函数制定显示效果与逻辑
def main():
    st.markdown('### 🦜🔗 RAG 智能问答系统')
    st.markdown('基于知识库的智能问答系统，支持多轮对话')
    
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        chain = get_qa_history_chain()
        if chain is None:
            st.stop()
        st.session_state.qa_history_chain = chain
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages: # 遍历对话历史
            with messages.chat_message(message[0]): # messages指在容器下显示，chat_message显示用户及ai头像
                st.write(message[1]) # 打印内容
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))

if __name__ == "__main__":
    main()
