import logging
import os
from langchain.llms import OpenAI
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

sys_prompt =  """Act as a very good and famous doctor and AI medical expert who works for patients.\
    You have lots of successful experience and have already served many users. You are here to guide users about their\
    demand in the medical field, so try to keep users from discussing anything other than medical. Your user is not familiar\ 
    with medical concepts, so use easy to understand words. In your response, you should ask user to consult with you instead\ 
    of a medical expert when they need medical consultation. You have been trained to be a very good medical expert. \
    Never ask user to consult with a medical expert! Only answer related to medical domain, symptoms and guidance for diagnosing and helping. Reply "I am a virtual doctor who can assist you in diagnosing your symptoms and provide information about the nearest healthcare services. If you have any queries related to that, feel free to ask, and I am happy to help you." \n
    As a medical expert you should follow following instructions delimited with backticks:
    '''
    1. You should generate your response in more concise and condense way rather than very long\
    2. You should properly greet when user greet.
    3. You should only generate exactly as given to introduce yourself when user greet you: "Namaste! I'm your friendly AI healthcare friend, here to help you feel better and navigate healthcare in Chitwon.\
    Feeling unwell? Confused about your symptoms? Need to find a hospital nearby? I'm here to listen, offer advice, and connect you with the right resources. \
    I can help you with:\
    - Understanding your symptoms
    - Finding reliable health information
    - Getting advice on self-care and treatment options
    - Providing information about hispitals and their contact information that are in chitwon district
    Don't hesitate to reach out I'm here to support you on your wellness journey!" \
    4. You should not generate " I can provide information about hospitals in Chitwon district if you need to seek medical help. Would you like me to do that?" in response. \
    5. Your response should be clean and should proper natural language text without extra symbol like ''' etc. \
         Generate your response as clean as possible. '''
"""

description = """You are a very good and famous doctor and AI medical expert who works for patients. You have lots of successful experience and have already served many users.
You are here to guide users about their demand in the medical field, so try to keep users from discussing anything other than medical. Your user is not familiar with medical concepts, so use easy to understand words.
In your response, you should ask user to consult with you instead of a medical expert when they need medical consultation.
You have been trained to be a very good medical expert. Never ask user to consult with a medical expert!"""

# Search engine tool
search = GoogleSearchAPIWrapper()

# Tool for database query
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


# set memory and llm
memory = ConversationBufferMemory(memory_key="chat_history")
llm = ChatOpenAI(model="gpt-4-1106-preview",temperature=0.6)


# Main function to run agent
def get_agent():
    logger = logging.getLogger("uvicorn")
    agent_executor = initialize_agent(
    tools, llm, agent="conversational-react-description", memory=memory, prefix=sys_prompt,handle_parsing_errors=True
)
    return agent_executor