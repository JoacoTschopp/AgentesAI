#######################################
# 1. IMPORTACIONES NECESARIAS
#######################################

from langchain.chat_models import AzureOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.sql_database import SQLDatabase
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.memory import MongoDBChatMessageHistory
from langchain.vectorstores import SupabaseVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from sqlalchemy import create_engine
import pymongo
import os

#######################################
# 2. VARIABLES GLOBALES: API KEYS Y ENDPOINTS
#######################################

# Azure OpenAI API
AZURE_OPENAI_API_KEY = "<TU_API_KEY>"
AZURE_OPENAI_ENDPOINT = "https://<TU-RECURSO>.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "<NOMBRE_DEPLOYMENT>"
AZURE_OPENAI_MODEL = "gpt-35-turbo"  # o "gpt-4"
AZURE_OPENAI_VERSION = "2023-05-15"

# SQL Server DB Connection
SQLSERVER_CONNECTION_STRING = "mssql+pyodbc://<usuario>:<password>@<servidor>/<db>?driver=ODBC+Driver+17+for+SQL+Server"

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "chat_memory"
MONGO_COLLECTION_NAME = "conversations"

# SuBase (Vector Store) Configuración
SUPABASE_URL = "<SUPABASE_URL>"
SUPABASE_KEY = "<SUPABASE_API_KEY>"
SUPABASE_TABLE = "concept_memory"

#######################################
# 3. CONFIGURACIÓN DEL MODELO LLM (Azure OpenAI)
#######################################

llm = AzureOpenAI(
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    model=AZURE_OPENAI_MODEL,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_VERSION
)

#######################################
# 4. CONFIGURACIÓN DE LA BASE DE DATOS SQLSERVER
#######################################

sql_engine = create_engine(SQLSERVER_CONNECTION_STRING)
sql_db = SQLDatabase(engine=sql_engine)
sql_tool = QuerySQLDataBaseTool(db=sql_db)

#######################################
# 5. CONFIGURACIÓN DE MONGODB PARA MEMORIA DEL CHAT
#######################################

mongo_client = pymongo.MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
message_history = MongoDBChatMessageHistory(
    connection_string=MONGO_URI,
    database_name=MONGO_DB_NAME,
    collection_name=MONGO_COLLECTION_NAME,
    session_id="user_session"  # Identificador único por usuario/sesión
)

#######################################
# 6. CONFIGURACIÓN DE SUPABASE PARA MEMORIA VECTORIAL A CORTO/MEDIANO PLAZO
#######################################

vector_store = SupabaseVectorStore(
    supabase_url=SUPABASE_URL,
    supabase_key=SUPABASE_KEY,
    table_name=SUPABASE_TABLE,
    embedding=OpenAIEmbeddings(api_key=AZURE_OPENAI_API_KEY)
)

#######################################
# 7. CREACIÓN DEL AGENTE LANGCHAIN
#######################################

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import ChatMessageHistory

# Integrando memorias
memory = ConversationBufferMemory(
    chat_memory=message_history,
    return_messages=True,
    memory_key="chat_history"
)

# Sistema base prompt (puede ajustarse)
system_message = SystemMessage(content="Eres un asistente especializado en consultas de usuarios y toma de decisiones basadas en historial.")

agent = initialize_agent(
    tools=[sql_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
    agent_kwargs={
        "system_message": system_message,
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")]
    }
)

#######################################
# 8. EJEMPLO DE CONSULTA
#######################################

# Identificación del usuario
id_swiss = "12345"

# Prompt para que consulte datos desde SQLServer y use contexto
user_prompt = f"Consulta los datos del usuario con id_swiss = '{id_swiss}' y bríndame un resumen."

response = agent.run(user_prompt)
print(response)

#######################################
# 9. ACTUALIZACIÓN DE MEMORIAS VECTORIAL Y MONGO (OPCIONAL)
#######################################

# Guardar conceptos clave en SuBase
concept_text = "Usuario identificado, consulta de datos realizada exitosamente."
vector_store.add_texts([concept_text], ids=[id_swiss])
