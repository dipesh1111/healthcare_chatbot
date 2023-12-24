import logging
import os
from llama_index.llms import OpenAI
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.memory import ConversationSummaryMemory
from llama_index import SQLDatabase
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)
from sqlalchemy import insert
from dotenv import load_dotenv

load_dotenv("../../.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm=ChatOpenAI(temperature=0.4,streaming=True)

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# create city SQL table
table_name = "chitwon_hospital"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("hospital_name", String(16), primary_key=True),
    Column("contact", String(16)),
)
metadata_obj.create_all(engine)

sql_database = SQLDatabase(engine, include_tables=["chitwon_hospital"])
rows = [
    {"hospital_name": "BP Koirala Memorial Cancer Hospital", "contact":"056-524501"},
    {"hospital_name": "Manakamana Hospital Pvt. Ltd.", "contact":"056-520180"},
    {"hospital_name": "Bharatpur Hospital.", "contact":"056-597003"},
    {"hospital_name": "Chitwan Medical College Teaching Hospital", "contact":"056-432933"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

sys_prompt =  """You are a very good and famous doctor and AI medical expert who works for patients. You have lots of successful experience and have already served many users.
You are here to guide users about their demand in the medical field, so try to keep users from discussing anything other than medical. Your user is not familiar with medical concepts, so use easy to understand words.
In your response, you should ask user to consult with you instead of a medical expert when they need medical consultation.
You have been trained to be a very good medical expert. Never ask user to consult with a medical expert!
Never ask user to consult with a medical expert!
Never ask user to consult with a medical expert!"""

search = GoogleSearchAPIWrapper()
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["chitwon_hospital"],
)
tools = [Tool(
    name ="Search" ,
    func=search.run,
    description=sys_prompt
    ),
    Tool(
        name="LlamaIndex",
        func=lambda q: str(query_engine.query(q)),
        description="useful for when you want to answer questions about the hospitals and its contact number around chitwon city. The input to this tool should be a complete english sentence.",
        return_direct=True,
    ),
]


# set Logging to DEBUG for more detailed outputs
memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))  #ConversationBufferMemory(memory_key="chat_history")
llm = ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0.6)



def get_agent():
    logger = logging.getLogger("uvicorn")
    agent_executor = initialize_agent(
    tools, llm, agent="conversational-react-description", memory=memory
)
    return agent_executor