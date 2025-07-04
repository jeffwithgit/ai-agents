from dotenv import load_dotenv  # 用于加载环境变量

load_dotenv()  # 加载.env文件中的环境变量
# 加载电商财报数据
from llama_index.core import SimpleDirectoryReader

A_docs = SimpleDirectoryReader(
    input_files=["电商B-Third Quarter 2023 Results.pdf"]
).load_data()
B_docs = SimpleDirectoryReader(
    input_files=["电商B-Third Quarter 2023 Results.pdf"]
).load_data()



# 从文档中创建索引
from llama_index.core import VectorStoreIndex
A_index = VectorStoreIndex.from_documents(A_docs)
B_index = VectorStoreIndex.from_documents(B_docs)

# 持久化索引（保存到本地）
from llama_index.core import StorageContext
A_index.storage_context.persist(persist_dir="./storage/A")
B_index.storage_context.persist(persist_dir="./storage/B")


# 从本地读取索引
from llama_index.core import load_index_from_storage
try:
    storage_context_A = StorageContext.from_defaults(
        persist_dir="./storage/A"
    )
    A_index = load_index_from_storage(storage_context_A)

    storage_context_B = StorageContext.from_defaults(
        persist_dir="./storage/B"
    )
    B_index = load_index_from_storage(storage_context_B)

    index_loaded = True
except:
    index_loaded = False


# 创建查询引擎
A_engine = A_index.as_query_engine(similarity_top_k=3)
B_engine = B_index.as_query_engine(similarity_top_k=3)


# 配置查询工具
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
query_engine_tools = [
    QueryEngineTool(
        query_engine=A_engine,
        metadata=ToolMetadata(
            name="A_Finance",
            description=(
                "用于提供A公司的财务信息 "
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=B_engine,
        metadata=ToolMetadata(
            name="B_Finance",
            description=(
                "用于提供B公司的财务信息 "
            ),
        ),
    ),
]


# 配置大模型
from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-4.1-2025-04-14")


# 创建ReAct Agent
from llama_index.core.agent import ReActAgent
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)


# 让Agent完成任务
agent.chat("比较一下两个公司的销售额")
