# imports
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from neo4j import Driver
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from docx import Document as DocxDocument
from pptx import Presentation
import openpyxl
from langchain.schema import Document
import gradio as gr # gradio for UI
from dotenv import load_dotenv

load_dotenv()

# Neo4j graph initialization
graph = Neo4jGraph()

# Dokumentumok már betöltve vannak
docs = []  # Add the documents that are already loaded here, they should be in the format of `Document` objects

# Neo4j graph setup - no need to reload data
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
llm_transformer = LLMGraphTransformer(llm=llm)

# Convert documents to graph format only if needed (we assume the graph is already populated)
graph_documents = llm_transformer.convert_to_graph_documents(docs)
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

# Neo4j Vector index retrieval (no need to rebuild the vector index)
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)
vector_retriever = vector_index.as_retriever()

driver = GraphDatabase.driver(
    uri=os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# Fulltext index query setup (no need to recreate the index)
def create_fulltext_index(tx):
    query = '''
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:__Entity__) 
    ON EACH [n.id];
    '''
    tx.run(query)

def create_index():
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
        print("Fulltext index created successfully.")

# If the index was already created, no need to create it again
# We are assuming the index and other data are already present in Neo4j

# Close the driver connection (we won't be creating new indexes here)
driver.close()

# Define Entities class for entity extraction
class Entities(BaseModel):
    """Identifying information about entities."""
    
    names: list[str] = Field(
        ..., description="All the person, organization, or business entities that appear in the text"
    )

# Define prompt for entity extraction
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting organization and person entities from the text."),
        ("human", "Use the given format to extract information from the following input: {question}"),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()

# Fulltext index query for retrieving data from Neo4j
def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question from the Neo4j database.
    """
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            WITH node
            CALL (node) {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result

# Fulltext index query for retrieving data from Neo4j
def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question from the Neo4j database.
    """
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            WITH node
            CALL (node) {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result

system_message = "You are a helpful assistant who answers the question based on only the provided context. Limit your answer purely on the context provided, do not add any additional information."
system_message += "After your answer please provide in a list the exact, complete sentences that you used from the context to answer the question."
system_message += "Always be accurate. If you don't know the answer, say so."
system_message += "Always provide detailed, comprehensive answers, including all relevant information from the context. It should be atleast 5 sentences."


# Ellenőrző függvény a messages lista validálására és javítására
def validate_and_fix_messages(messages):
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{i}] nem dict típusú: {message}")
        if "role" not in message or "content" not in message:
            raise ValueError(f"messages[{i}] hiányos: {message}")
        if not isinstance(message["role"], str):
            raise ValueError(f"messages[{i}]['role'] nem sztring: {message['role']}")
        if not isinstance(message["content"], str):
            # Ha a content nem sztring, próbálja meg javítani
            if isinstance(message["content"], list):
                # Lista esetén összefűzi sztringgé
                messages[i]["content"] = " ".join(map(str, message["content"]))
            else:
                # Egyéb esetben átalakítja sztringgé
                messages[i]["content"] = str(message["content"])
    return messages

# Frissített full_retriever függvény
def full_retriever(question: str):
    graph_data = graph_retriever(question)
    vector_data = [
        (
            el.page_content,
            el.metadata.get("source", "Ismeretlen forrás"),
            el.metadata.get("page", "N/A")
        )
        for el in vector_retriever.invoke(question)
    ]

    # Gyűjtsük össze a használt mondatokat, elkerülve az ismétléseket
    unique_sentences = list(dict.fromkeys(vector_data))

    # A mondatok előtt jelenjen meg a hely és oldalszám
    vector_context = "\n".join(
        f"Forrás: {source} (Oldal: {page})\n{content}"
        for content, source, page in unique_sentences
    )

    # Egyedi mondatok referencialistája, összekapcsolva a forrásukkal
    sentence_references = [
        f"{idx + 1}. Forrás: {source} - Oldal: {page}\nMondat: {content}"
        for idx, (content, source, page) in enumerate(unique_sentences)
    ]

    final_data = f"""Graph data:
{graph_data}
Vector data:
{vector_context}
    """

    return final_data, sentence_references

# A chat függvény frissítése
def chat(message, history):
    additional_context, sentence_references = full_retriever(message)

    context_message = f"{system_message}\n\nAdditional Context:\n{additional_context}"

    messages = [{"role": "system", "content": context_message}] + history + [{"role": "user", "content": message}]

    # Ellenőrzés és javítás a messages lista elküldése előtt
    try:
        messages = validate_and_fix_messages(messages)
    except ValueError as e:
        return f"Hiba a messages lista ellenőrzése közben: {e}"

    response = openai.chat.completions.create(model=MODEL, messages=messages)
    chat_response = response.choices[0].message.content

    if sentence_references:
        # Pontosan kapcsolódó referenciák kiírása
        sentence_references_message = "\n".join(sentence_references)
        chat_response += f"\n\nVálasz: {sentence_references_message}"
    else:
        chat_response += "\n\nForrások: Nincsenek elérhető dokumentumok."

    return chat_response

# Gradio UI setup for chat
gr.ChatInterface(fn=chat, type="messages").launch(share=True)




from openai import OpenAI
MODEL = "gpt-4o-mini"
openai = OpenAI()



# Document summarization function (using already loaded docs)
summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are summarizing documents into a concise, coherent paragraph."),
        ("human", "Summarize the following document: {document_text}"),
    ]
)

summary_chain = summary_prompt | llm

def summarize_documents(docs):
    summaries = []
    for doc in docs:
        summary = summary_chain.invoke({"document_text": doc.page_content})
        summaries.append(f"File: {doc.metadata['source']}\nSummary: {summary}\n")
    return "\n".join(summaries)

# Dokumentumösszefoglalók generálása
summaries = summarize_documents(docs)
print(f"Dokumentumok összefoglalója:\n{summaries}")

# Gradio UI for document summaries
def display_summaries():
    return summaries

gr.Interface(
    fn=display_summaries,
    inputs=None,
    outputs="text",
    title="Dokumentum Összefoglaló"
).launch(share=True)
