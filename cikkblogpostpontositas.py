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

    # Csak az egyedi mondatokat tartjuk meg a forrással és oldallal együtt
    unique_sentences = []
    seen_content = set()
    for content, source, page in vector_data:
        if content not in seen_content:
            seen_content.add(content)
            unique_sentences.append((content, source, page))

    vector_context = "\n".join(
        f"#Document: {source}\n{content}"
        for content, source, _ in unique_sentences
    )

    final_data = f"""Graph data:
{graph_data}
Vector data:
{vector_context}
"""

    return final_data, unique_sentences 

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

def normalize_text(text):
    """Normalize text for better comparison"""
    import re
    # Remove the 'text:' prefix if exists
    text = re.sub(r'^text:\s*', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but preserve important ones
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    # Split into words and sort them for better partial matching
    words = text.split()
    return ' '.join(sorted(words))


def calculate_similarity(text1, text2):
    """Calculate similarity between two texts using multiple metrics"""
    from difflib import SequenceMatcher
    from collections import Counter
    
    # Normalize texts
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    # Calculate word-level similarity
    words1 = norm1.split()
    words2 = norm2.split()
    
    # Create word frequency counters
    counter1 = Counter(words1)
    counter2 = Counter(words2)
    
    # Calculate word overlap
    common_words = set(counter1.keys()) & set(counter2.keys())
    if not common_words:
        return 0.0
    
    # Calculate weighted word overlap
    total_weight = 0
    common_weight = 0
    
    for word in set(words1 + words2):
        count1 = counter1[word]
        count2 = counter2[word]
        weight = max(count1, count2)
        total_weight += weight
        if word in common_words:
            common_weight += min(count1, count2)
    
    word_similarity = common_weight / total_weight if total_weight > 0 else 0
    
    # Calculate sequence similarity for word order
    sequence_sim = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Calculate length ratio penalty
    len_ratio = min(len(words1), len(words2)) / max(len(words1), len(words2))
    
    # Combine metrics with weights
    final_similarity = (
        word_similarity * 0.5 +  # Word overlap is most important
        sequence_sim * 0.3 +     # Sequence matching helps with order
        len_ratio * 0.2         # Length ratio prevents matching very different sized texts
    )
    
    return final_similarity


def find_best_source_match(sentence, unique_sentences, threshold=0.25):
    """Find the best matching source using improved similarity metrics"""
    best_match = {
        'score': 0,
        'source': None,
        'exact_match': False,
        'matching_content': None
    }
    
    # Clean the input sentence
    clean_sentence = sentence.strip('" ')
    
    for content, source, page in unique_sentences:
        # Split content into sentences for more granular matching
        content_sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        for content_sentence in content_sentences:
            # Calculate similarity
            similarity = calculate_similarity(clean_sentence, content_sentence)
            
            # Update best match if better score found
            if similarity > best_match['score']:
                best_match = {
                    'score': similarity,
                    'source': source,
                    'exact_match': similarity > 0.85,
                    'matching_content': content_sentence,
                    'page': page
                }
    
    return best_match if best_match['score'] >= threshold else None

def extract_used_sentences(chat_response):
    """Extract used sentences from the chat response with improved format handling"""
    
    # Define all possible section headers and their variations
    headers = [
        "Sentences used:",
        "Sentences used from the context:",
        "The exact sentences used:",
        "Used sentences:",
        "The following sentences were used:"
    ]
    
    # Find the first instance of any header
    start_index = -1
    used_header = None
    
    for header in headers:
        if header in chat_response:
            index = chat_response.find(header)
            if start_index == -1 or index < start_index:
                start_index = index
                used_header = header
    
    if start_index == -1:
        return chat_response, None, None
        
    # Split the response
    main_response = chat_response[:start_index].strip()
    sentences_part = chat_response[start_index + len(used_header):].strip()
    
    # Find where the source attributions begin (if they exist)
    attribution_markers = ["Forrás attribúciók:", "Sources:", "Source attributions:"]
    end_index = float('inf')
    
    for marker in attribution_markers:
        idx = sentences_part.find(marker)
        if idx != -1 and idx < end_index:
            end_index = idx
            
    if end_index != float('inf'):
        sentences_part = sentences_part[:end_index].strip()
    
    # Parse sentences with improved quote handling
    sentences = []
    current_sentence = ""
    
    for line in sentences_part.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for numbered sentence patterns
        if (line[0].isdigit() and '. "' in line) or (line[0].isdigit() and '. ' in line):
            if current_sentence:
                sentences.append(current_sentence.strip())
            current_sentence = line
        else:
            current_sentence += " " + line
            
    if current_sentence:
        sentences.append(current_sentence.strip())
    
    # Clean up sentences and remove any duplicate attributions
    cleaned_sentences = []
    for sentence in sentences:
        # Remove any source attributions that might have been included
        if " - Source:" in sentence:
            sentence = sentence.split(" - Source:")[0]
        # Remove any leading numbers and dots
        if sentence[0].isdigit():
            parts = sentence.split('. ', 1)
            if len(parts) > 1:
                sentence = parts[1]
        cleaned_sentences.append(sentence.strip())
    
    return main_response, cleaned_sentences, used_header

def chat(message, history):
    print("\n=== Starting new chat interaction ===")
    print(f"User message: {message}")
    
    additional_context, unique_sentences = full_retriever(message)
    print(f"\nReceived {len(unique_sentences)} unique sentences from retriever")
    
    context_message = f"{system_message}\n\nAdditional Context:\n{additional_context}"
    messages = [{"role": "system", "content": context_message}] + history + [{"role": "user", "content": message}]
    
    try:
        messages = validate_and_fix_messages(messages)
    except ValueError as e:
        return f"Error validating messages: {e}"
    
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    chat_response = response.choices[0].message.content
    
    # Extract sentences using improved function
    main_response, used_sentences, header = extract_used_sentences(chat_response)
    
    if not used_sentences:
        return chat_response + "\n\nForrások: A válasz nem tartalmaz idézett mondatokat."
    
    # Process each sentence and find sources - only once
    sentence_references = []
    all_sources = set()
    
    print("\nProcessing used sentences:")
    for i, sentence in enumerate(used_sentences, 1):
        try:
            # Extract the quoted text
            quoted_text = None
            if '"' in sentence:
                start = sentence.find('"') + 1
                end = sentence.rfind('"')
                if start > 0 and end > start:
                    quoted_text = sentence[start:end]
            else:
                quoted_text = sentence
            
            if not quoted_text:
                continue
                
            print(f"\nLooking for source for: {quoted_text}")
            best_match = find_best_source_match(quoted_text, unique_sentences)
            
            if best_match:
                print(f"Found match in {best_match['source']} with score {best_match['score']}")
                all_sources.add(best_match['source'])
                confidence = "Pontos egyezés" if best_match['exact_match'] else f"Hasonlósági pontszám: {best_match['score']:.2f}"
                page_info = f" (oldal: {best_match.get('page', 'N/A')})" if best_match.get('page') != "N/A" else ""
                sentence_references.append(
                    f"A(z) {i}. mondat forrása:\n"
                    f"- {best_match['source']}{page_info} - {confidence}\n"
                    f"  Eredeti szöveg: \"{best_match['matching_content']}\""
                )
            else:
                print(f"No match found for sentence {i}")
                sentence_references.append(f"A(z) {i}. mondathoz nem található forrás az adatbázisban")
                
        except Exception as e:
            print(f"Error processing sentence {i}: {str(e)}")
            sentence_references.append(f"A(z) {i}. mondathoz hiba történt a forrás keresése közben: {str(e)}")
    
    # Generate final response with single attribution
    final_response = (
        f"{main_response}\n\n"
        f"Sentences used:\n{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(used_sentences))}\n\n"
        f"Forrás attribúciók:\n{chr(10).join(sentence_references)}\n\n"
        f"Összes felhasznált forrás:\n" + 
        "\n".join(f"- {source}" for source in sorted(all_sources))
    )
    
    return final_response

# Helper function to print the contents of unique_sentences for debugging
def debug_print_unique_sentences(unique_sentences):
    print("\nDEBUG: Contents of unique_sentences:")
    for idx, (content, source, page) in enumerate(unique_sentences, 1):
        print(f"\n=== Document {idx} ===")
        print(f"Source: {source}")
        print(f"Page: {page}")
        print(f"Content preview: {content[:200]}...")  # Print first 200 characters of content


from openai import OpenAI
MODEL = "gpt-4o-mini"
openai = OpenAI()

system_message = "You are a helpful assistant who answers the question based on only the provided context. Limit your answer purely on the context provided, do not add any additional information."
system_message += "After your answer please provide in a list the exact, complete sentences that you used from the context to answer the question."
system_message += "Always be accurate. If you don't know the answer, say so."
system_message += "Always provide detailed, comprehensive answers, including all relevant information from the context. It should be atleast 5 sentences."



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


article_system_message = """
Elemezd a megadott forrásokat és stílusukat, majd írj egy új blog posztot a megadott témáról, amely:

1. TARTALMI KÖVETELMÉNYEK:
   - Tökéletesen követi a forrásanyagok stílusát és hangnemét
   - A kért témához kapcsolódó, de új tartalmat generál
   - Hasonló hosszúságú, mint a forrásanyagok átlagos hosszúsága
   - A forrásokból vett szakszókincset és fogalmakat használja
   - Megtartja a források jellegzetes kifejezésmódját

2. BLOG-SPECIFIKUS ELEMEK:
   - Figyelemfelkeltő címet és bevezetést tartalmaz
   - Személyes hangvételű, közvetlen stílusú
   - Interaktív elemeket (kérdések, felvetések) tartalmaz
   - Gyakorlati példákkal és alkalmazásokkal színesíti a tartalmat

3. STRUKTÚRA:
   - Logikusan felépített, jól tagolt szerkezet
   - Alcímekkel és bekezdésekkel tagolt
   - Kulcspontok kiemelése
   - Konklúzió vagy call-to-action a végén

A végeredmény egy koherens, olvasóbarát blog poszt legyen, amely ötvözi a forrásanyagok szakmaiságát 
a blog műfaj közvetlenségével és olvashatóságával.
"""

def generate_article(topic, target_audience="general", include_seo=True, include_visual=True, sample_size=5):
    print(f"\n=== Generating enhanced blog post about: {topic} ===")
    
    # Retrieve relevant documents
    additional_context, unique_sentences = full_retriever(topic)
    
    # Sample collection logic remains the same
    import random
    sample_texts = []
    sample_sources = set()
    
    if unique_sentences:
        samples = random.sample(unique_sentences, min(sample_size, len(unique_sentences)))
        for content, source, _ in samples:
            sample_texts.append(content)
            sample_sources.add(source)
    
    style_samples = "\n\n".join(sample_texts)
    
    # Define the system messages and prompts that were missing
    enhanced_blog_system_message = """
    Elemezd a megadott forrásokat és stílusukat, majd írj egy jól strukturált blog posztot, amely:

    1. SZERKEZETI ELEMEK:
       - Figyelemfelkeltő cím (főcím és alcím kombinációja)
       - Bevezető hook, ami felkelti az érdeklődést
       - Világosan tagolt alfejezetek
       - Összefoglaló vagy call-to-action a végén
       - Belső linkek és hivatkozási javaslatok
    
    2. TARTALOM ÉS STÍLUS:
       - Követi a mintadokumentumok stílusát
       - Szakmai, de közérthető megfogalmazás
       - Logikus gondolatmenet
       - Gyakorlati példák és alkalmazások
    """

    content_prompt = f"""
    Téma: {topic}
    Célközönség: {target_audience}

    Stílusminta dokumentumok:
    {style_samples}

    Kérlek, írj egy részletes blog posztot a megadott téma alapján.
    """

    # Define the visual content prompt that was missing
    visual_prompt = f"""
    A következő blog poszthoz készíts részletes vizuális tartalom javaslatokat:

    Téma: {topic}
    Célközönség: {target_audience}

    Kérlek, adj javaslatokat:
    1. Képek és Illusztrációk:
       - Főkép javaslat és leírás
       - Kiegészítő képek javaslatai
    2. Infografikák:
       - Témához kapcsolódó infografika javaslatok
       - Kulcspontok vizualizációs ötletei
    3. Design elemek:
       - Tördelési javaslatok
       - Kiemelések, blokkok elrendezése
    """

    visual_messages = [
        {"role": "system", "content": "Te egy vizuális tartalom és design szakértő vagy."},
        {"role": "user", "content": visual_prompt}
    ]

    # Define the SEO prompt that was missing
    seo_prompt = f"""
    Készíts SEO elemzést a következő blog poszt témához:
    {topic}

    Add meg:
    1. Kulcsszavak
    2. Meta leírás javaslatok
    3. Címoptimalizálási javaslatok
    4. Belső linkek javaslatai
    """

    seo_messages = [
        {"role": "system", "content": "Te egy SEO és tartalomoptimalizálási szakértő vagy."},
        {"role": "user", "content": seo_prompt}
    ]
    
    try:
        # Generate main content
        content_messages = [
            {"role": "system", "content": enhanced_blog_system_message},
            {"role": "user", "content": content_prompt}
        ]
        
        content_response = openai.chat.completions.create(
            model=MODEL, 
            messages=content_messages,
            temperature=0.7,
            max_tokens=2000
        )
        blog_content = content_response.choices[0].message.content
        
        final_output = f"# Blog Poszt\n\n{blog_content}\n"
        
        # Only include SEO analysis if requested
        if include_seo:
            seo_response = openai.chat.completions.create(
                model=MODEL,
                messages=seo_messages,
                temperature=0.7,
                max_tokens=1000
            )
            seo_content = seo_response.choices[0].message.content
            final_output += f"\n## SEO és Kulcsszó Elemzés\n\n{seo_content}\n"
        
        # Only include visual suggestions if requested
        if include_visual:
            visual_response = openai.chat.completions.create(
                model=MODEL,
                messages=visual_messages,
                temperature=0.7,
                max_tokens=800
            )
            visual_content = visual_response.choices[0].message.content
            final_output += f"\n## Vizuális Tartalom Javaslatok\n\n{visual_content}\n"
        
        # Add sources
        sources_text = "\n".join([f"- {src}" for src in sample_sources])
        final_output += f"\n## Források\nA blog poszt a következő dokumentumok stílusának felhasználásával készült:\n{sources_text}"
        
        return final_output
        
    except Exception as e:
        return f"Hiba történt a blog poszt generálása közben: {str(e)}"

# New Gradio UI implementation
def create_blog_ui():
    with gr.Blocks(title="AI Blog Poszt Generátor", theme=gr.themes.Soft()) as blog_interface:
        gr.Markdown("""
        # 📝 AI Blog Poszt Generátor
        Készítsen professzionális, SEO-optimalizált blog posztokat egyszerűen!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                topic_input = gr.Textbox(
                    label="Blog Poszt Témája",
                    placeholder="Írja be a blog poszt témáját...",
                    lines=2
                )
                
                with gr.Row():
                    audience = gr.Radio(
                        choices=["general", "expert", "intermediate"],
                        value="general",
                        label="Célközönség",
                        info="Válassza ki a célközönséget a megfelelő stílus érdekében"
                    )
                
                with gr.Row():
                    generate_btn = gr.Button("💫 Generálás", variant="primary")
                    clear_btn = gr.Button("🗑️ Törlés")
            
            with gr.Column(scale=3):
                output = gr.Markdown(
                    label="Generált Blog Poszt",
                    show_label=True
                )
        
        with gr.Accordion("✨ Haladó Beállítások", open=False):
            with gr.Row():
                include_seo = gr.Checkbox(
                    label="SEO Elemzés",
                    value=True,
                    info="SEO javaslatok és kulcsszó elemzés generálása"
                )
                include_visual = gr.Checkbox(
                    label="Vizuális Javaslatok",
                    value=True,
                    info="Képek és design elemek javaslatainak generálása"
                )
        
        with gr.Accordion("ℹ️ Útmutató", open=False):
            gr.Markdown("""
            ### Hogyan használja a generátort?
            1. **Téma megadása**: Írja be a kívánt blog poszt témáját
            2. **Célközönség választása**:
               - General: Általános közönség
               - Expert: Szakértői szint
               - Intermediate: Középhaladó szint
            3. **Haladó beállítások**:
               - SEO elemzés: Kulcsszavak és metaleírások
               - Vizuális javaslatok: Képek és design elemek
            
            ### Tippek a jobb eredményért:
            - Legyen konkrét a téma megadásánál
            - Válassza ki a megfelelő célközönséget
            - Használja a haladó beállításokat részletesebb outputért
            """)
        
        # Event handlers
        generate_btn.click(
            fn=lambda topic, audience, seo, visual: generate_article(
                topic,
                target_audience=audience,
                include_seo=seo,
                include_visual=visual
            ),
            inputs=[topic_input, audience, include_seo, include_visual],
            outputs=output
        )
        
        clear_btn.click(
            fn=lambda: (None, "general", True, True),
            inputs=None,
            outputs=[topic_input, audience, include_seo, include_visual]
        )
        
        return blog_interface

# Replace the existing Gradio interface with the new one
if __name__ == "__main__":
    blog_ui = create_blog_ui()
    blog_ui.launch(share=True)