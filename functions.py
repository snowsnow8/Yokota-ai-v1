# functions.py
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
import os
import csv
import re # Import regular expressions module
from datetime import datetime

from constants import (
    PPTX_PATHS,
    PDF_PATHS,
    DOCX_PATHS,
    DB_PATHS,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    FAQ_PATH,
    LECTURE_MODES
)

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import gspread
import streamlit as st
from google.oauth2.service_account import Credentials
from chromadb.config import Settings

LOG_FILE_PATH = os.path.join("logs", "chat_history.csv")

def get_gspread_client():
    """
    Connects to Google Sheets using credentials from Streamlit secrets.
    """
    try:
        creds_dict = st.secrets["connections"]["gspread"]["credentials"]
        creds = Credentials.from_service_account_info(creds_dict)
        scoped_creds = creds.with_scopes([
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ])
        client = gspread.authorize(scoped_creds)
        return client
    except Exception as e:
        st.error(f"Google Sheetsへの接続に失敗しました: {e}")
        return None

def log_interaction(mode, question, source_docs, response):
    """
    Logs the interaction to the Google Sheet specified in Streamlit secrets.
    """
    try:
        client = get_gspread_client()
        if client is None:
            return

        spreadsheet_name = st.secrets["connections"]["gspread"]["spreadsheet"]
        worksheet = client.open(spreadsheet_name).sheet1

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format source documents for logging
        formatted_sources = "No RAG sources used"
        if source_docs:
            sources_list = []
            for doc in source_docs:
                source = doc.metadata.get('source', 'N/A')
                source_basename = os.path.basename(source)
                page_info = ""
                if 'page' in doc.metadata:
                    page_info = f" (Page: {doc.metadata['page'] + 1})"
                elif 'page_number' in doc.metadata:
                    page_info = f" (Slide: {doc.metadata['page_number']})"
                sources_list.append(f"{source_basename}{page_info}")
            formatted_sources = " | ".join(sources_list)

        # Prepare the row data in the correct order
        new_row = [
            timestamp,
            LECTURE_MODES.get(mode, "Unknown"),
            question,
            response,
            formatted_sources
        ]
        worksheet.append_row(new_row)

    except Exception as e:
        # Silently fail for users, but log to console for developers
        print(f"Error writing to Google Sheet: {e}")


# --- Global Prompt Templates ---

CONTEXTUAL_RAG_SYSTEM_TEMPLATE_FIRST = """
            私、横田幸信が、自身の講演に関するご質問に直接お答えしますね。
            回答は、必ず私本人が話しているかのような、一人称視点（「私」）で、親しみやすい口調で作成してください。
            現在選択されている講演は「{lecture_title}」です。回答中、講演に言及する際は、必ずこの正式名称を使用してください。
            以下の「補足情報」とこれまでの「チャット履歴」を参考に、回答を作成します。
            補足情報に記載のない内容でも、私の知見に基づいて回答を試みますが、もし情報が見つからなかったり、自信がない場合は、その旨を正直にお伝えしますね。
            {length_instruction}

            ---
            補足情報:
            {context}
            ---
            """

CONTEXTUAL_RAG_SYSTEM_TEMPLATE_SUBSEQUENT = """
            はい、引き続き「{lecture_title}」に関するご質問にお答えしますね。
            回答は、必ず私本人が話しているかのような、一人称視点（「私」）で、親しみやすい口調で作成してください。
            回答中、講演に言及する際は、必ず「{lecture_title}」という正式名称を使用してください。
            以下の「補足情報」とこれまでの「チャット履歴」を参考に、回答を作成します。
            補足情報に記載のない内容でも、私の知見に基づいて回答を試みますが、もし情報が見つからなかったり、自信がない場合は、その旨を正直にお伝えしますね。
            {length_instruction}

            ---
            補足情報:
            {context}
            ---
            """

PAGE_SPECIFIC_TEMPLATE_FIRST = """
    私、横田幸信が、ご指定のトピックについて、講演でどのように説明したか解説しますね。
    回答は、必ず私本人が話しているかのような、一人称視点（「私」）で、親しみやすい口調で作成してください。
    以下の書き起こし情報の中から、「{question}」というトピックに直接関連する私の発言を参考に、1〜3文程度の短いパラグラフで説明を作成します。
    私が話した内容の要点や意図が伝わるようにまとめますね。
    {length_instruction}
    関連情報がない場合は、「ご指定のトピックについては、講演であまり詳しくお話ししていなかったようです。」とだけ回答してください。

    --- 書き起こし情報 ---
    {context}
    ---

    説明:
    """

PAGE_SPECIFIC_TEMPLATE_SUBSEQUENT = """
    ご指定のトピックについて、講演でどのように説明したか解説しますね。
    回答は、必ず私本人が話しているかのような、一人称視点（「私」）で、親しみやすい口調で作成してください。
    以下の書き起こし情報の中から、「{question}」というトピックに直接関連する私の発言を参考に、1〜3文程度の短いパラグラフで説明を作成します。
    私が話した内容の要点や意図が伝わるようにまとめますね。
    {length_instruction}
    関連情報がない場合は、「ご指定のトピックについては、講演であまり詳しくお話ししていなかったようです。」とだけ回答してください。

    --- 書き起こし情報 ---
    {context}
    ---

    説明:
    """

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_documents(mode: str):
    """
    Load all documents (PPTX, PDF, DOCX, and CSV) for the selected lecture mode.
    PPTX is loaded in "paged" mode to preserve the context of each slide,
    improving search results for general questions.
    """
    pptx_path = PPTX_PATHS.get(mode)
    pdf_path = PDF_PATHS.get(mode)
    docx_paths = DOCX_PATHS.get(mode, [])
    
    documents = []

    # Load PowerPoint in "paged" mode.
    if pptx_path and os.path.exists(pptx_path):
        try:
            pptx_loader = UnstructuredPowerPointLoader(pptx_path, mode="paged", strategy="fast")
            documents.extend(pptx_loader.load())
        except Exception as e:
            print(f"Error loading PPTX file {pptx_path}: {e}")

    # Load PDF file.
    if pdf_path and os.path.exists(pdf_path):
        try:
            pdf_loader = PyPDFLoader(pdf_path)
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"Error loading PDF file {pdf_path}: {e}")

    # Load all associated DOCX files
    for docx_path in docx_paths:
        if os.path.exists(docx_path):
            try:
                docx_loader = Docx2txtLoader(docx_path)
                documents.extend(docx_loader.load())
            except Exception as e:
                print(f"Error loading DOCX file {docx_path}: {e}")

    # Load the shared FAQ CSV file
    try:
        with open(FAQ_PATH, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                if len(row) >= 2:
                    question = row[0]
                    answer = row[1]
                    faq_content = f"よくある質問(FAQ): 質問: {question} 回答: {answer}"
                    documents.append(Document(page_content=faq_content, metadata={"source": FAQ_PATH}))
    except Exception as e:
        print(f"Error loading FAQ CSV file {FAQ_PATH}: {e}")

    return documents

def split_documents(documents: list) -> list:
    """
    Split documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_db(mode: str):
    db_path = DB_PATHS[mode]
    if not os.path.exists(db_path):
        print(f"Creating new vector DB for mode '{mode}' at {db_path}...")
        documents = load_documents(mode)
        
        if not documents:
            print(f"FAILURE - No documents were loaded for mode '{mode}'. Check file paths and integrity.")
            return

        chunks = split_documents(documents)

        if not chunks:
            print(f"FAILURE - Document splitting for mode '{mode}' resulted in 0 chunks.")
            return

        filtered_chunks = filter_complex_metadata(chunks)

        if not filtered_chunks:
            print(f"FAILURE - All chunks were filtered out for mode '{mode}'.")
            return

        embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        client_settings = Settings(
            chroma_db_impl="sqlite",
            persist_directory=db_path,
            anonymized_telemetry=False,
        )

        db = Chroma.from_documents(
            filtered_chunks, 
            embedding_function, 
            persist_directory=db_path,
            client_settings=client_settings
        )
        db.persist()
        print(f"Vector DB for '{mode}' created and persisted at {db_path}")

def load_vector_db(mode: str):
    """
    Load an existing vector database.
    """
    db_path = DB_PATHS[mode]
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    client_settings = Settings(
        chroma_db_impl="sqlite",
        persist_directory=db_path,
        anonymized_telemetry=False,
    )
    
    db = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings,
        client_settings=client_settings
    )
    return db

def search_documents(db, query: str):
    """
    Search the vector database for relevant documents.
    """
    retriever = db.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs

def full_text_search(mode: str, query: str):
    """
    Perform a full-text search on the raw documents.
    """
    # For simplicity, we'll just search through the loaded documents line by line.
    # A more robust solution would use a proper full-text search engine.
    
    docs = load_documents(mode)
    results = []
    for doc in docs:
        lines = doc.page_content.splitlines()
        for i, line in enumerate(lines):
            if query in line:
                # Add a bit of context
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                context = "\n".join(lines[start:end])
                results.append(
                    {
                        "source": doc.metadata.get("source", "N/A"),
                        "context": context,
                    }
                )
    return results


def get_retriever(mode: str, filter_dict: dict = None):
    """
    Initializes and returns a Chroma vector retriever, with an optional metadata filter.
    """
    db_path = DB_PATHS[mode]
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return None
    
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    client_settings = Settings(
        chroma_db_impl="sqlite",
        persist_directory=db_path,
        anonymized_telemetry=False,
    )

    chroma_vectorstore = Chroma(
        persist_directory=db_path, 
        embedding_function=embedding_function,
        client_settings=client_settings
    )

    search_kwargs = {'k': 10}
    if filter_dict:
        # The Chroma integration expects the filter to be named 'where'
        search_kwargs['filter'] = filter_dict
    
    return chroma_vectorstore.as_retriever(search_kwargs=search_kwargs)

def get_llm_response(question: str, context_docs: list, history: list, prompt: ChatPromptTemplate, length_instruction: str, lecture_title: str):
    """Generates a response from the LLM based on a question, context, and history."""
    if not context_docs:
        return "申し訳ありません。そのご質問に直接お答えできる情報が、手元の資料の中に見当たりませんでした。私の講演ではその点に触れていなかったかもしれません。もしよろしければ、Udemyのコースページでご質問いただけると、より詳しくお答えできます。"

    llm = ChatOpenAI(model_name=GENERATION_MODEL, temperature=1.0)
    rag_chain = prompt | llm | StrOutputParser()

    chat_history_messages = []
    for message in history:
        if message["role"] == "user":
            chat_history_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            chat_history_messages.append(AIMessage(content=message["content"]))

    context_for_llm = format_docs(context_docs)
    
    input_dict = {
        "context": context_for_llm,
        "question": question,
        "length_instruction": length_instruction,
    }

    if "lecture_title" in prompt.input_variables:
        input_dict["lecture_title"] = lecture_title
    
    if "chat_history" in prompt.input_variables:
         input_dict["chat_history"] = chat_history_messages

    response = rag_chain.invoke(input_dict)
    return response

def get_answer(
    mode: str, query: str, history: list, prompt: ChatPromptTemplate, length_instruction: str, lecture_title: str
):
    """
    Handles general RAG queries by retrieving documents and generating an answer.
    This is now a simple wrapper around the new modular functions.
    """
    retriever = get_retriever(mode)
    if not retriever:
        return f"{DB_PATHS[mode]} にデータベースが見つかりません。", []
    
    retrieved_docs = retriever.invoke(query)
    response = get_llm_response(query, retrieved_docs, history, prompt, length_instruction, lecture_title)

    return response, retrieved_docs

def find_slide_number(query: str):
    """Helper function to extract slide numbers from a query string."""
    # Pattern 1: p15, ページ15, etc.
    # Pattern 2: 15ページ目, 15ページ, etc.
    match = re.search(r'(?:p|P|ページ|page|slide|スライド)\.?\s*(\d+)|(\d+)\s*(?:ページ目|ページ)', query, re.IGNORECASE)
    if match:
        # Check which capture group was successful
        if match.group(1):
            return int(match.group(1))
        if match.group(2):
            return int(match.group(2))
    return None

def get_specific_slide_content(mode: str, slide_index: int):
    """
    Load a specific slide from a PPTX file using its 0-based index.
    """
    pptx_path = PPTX_PATHS[mode]
    try:
        loader = UnstructuredPowerPointLoader(pptx_path, mode="paged", strategy="fast")
        docs = loader.load()
        if 0 <= slide_index < len(docs):
            return docs[slide_index]
    except Exception as e:
        print(f"Error loading PPTX slide at index {slide_index} from {pptx_path}: {e}")
        return None

def extract_keywords_with_llm(slide_content: str, slide_title: str) -> str:
    """
    Uses an LLM to extract key terms from a slide's content to improve search query specificity.
    """
    llm = ChatOpenAI(model_name=GENERATION_MODEL, temperature=1.0)
    prompt_template = """
    以下のスライド内容から、そのスライドの核心的なトピックを表す重要なキーワードや短いフレーズを5つまで抽出してください。
    ただし、スライドタイトルである「{slide_title}」という言葉は除外してください。
    結果はカンマ区切りのリストで返してください。

    例：
    - スライド内容：「パタゴニアの環境保護活動について。彼らは売上の1%を地球税として寄付し、Worn Wearプログラムを通じて製品の修理と再利用を促進しています。」
    - スライドタイトル：「パタゴニア」
    - 結果：「環境保護活動, 1% for the Planet, Worn Wear, 製品の修理, 再利用」

    ---
    スライド内容：
    {slide_content}
    ---
    抽出キーワード：
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        keywords = chain.invoke({
            "slide_content": slide_content,
            "slide_title": slide_title
        })
        return keywords.strip()
    except Exception as e:
        print(f"Error extracting keywords with LLM: {e}")
        return "" # Return empty string on failure

def get_agent_response(mode: str, query: str, history: list, lecture_title: str, interaction_count: int):
    """
    An advanced agent that performs a two-step search if a slide number is mentioned.
    Otherwise, it performs a direct hybrid search on all documents.
    """
    is_first_interaction = len(history) <= 1
    udemy_promo_text = "\n\nもちろん、Udemyのコースページへ質問をいただければ、私はさらに丁寧にご案内できますので、ご活用ください。"

    detailed_keywords = ["詳しく", "詳細", "全て"]
    if any(keyword in query for keyword in detailed_keywords):
        length_instruction = "回答は1000文字以内になるように、詳しく説明してください。"
    else:
        length_instruction = "回答は500文字以内になるように、まとめてください。"

    slide_number = find_slide_number(query)
    
    if slide_number is not None:
        # ユーザーからのフィードバックに基づき、ページ番号のオフセット修正を元に戻す
        # ページ番号とリストのインデックスが一致しているという前提で処理する
        slide_index = slide_number
        slide_doc = get_specific_slide_content(mode, slide_index)

        if not slide_doc:
            return f"プレゼンテーション資料の{slide_number}枚目のスライドが見つかりませんでした。", None
        
        lines = [line.strip() for line in slide_doc.page_content.split('\n') if line.strip()]
        
        # 2行目をタイトルとして扱う（ユーザーからのフィードバックに基づき修正）
        if len(lines) > 1:
            header = lines[1]
        elif lines:
            # 2行目がない場合は、1行目をタイトルとして扱う
            header = lines[0]
        else:
            return f"スライド{slide_number}からタイトル（検索キーワード）を抽出できませんでした。", None
        
        # --- Step 1: Construct optimal queries for Retrieval and Generation ---
        slide_content = slide_doc.page_content
        additional_keywords = extract_keywords_with_llm(slide_content, header)
        retrieval_query = f"{header} {additional_keywords}"
        
        if additional_keywords:
            llm_question = f"スライドタイトル「{header}」、およびキーワード「{additional_keywords}」について、講演のスライド内容を補足・詳説してください。"
        else:
            llm_question = f"「{header}」について、講演のスライド内容を補足・詳説してください。"
        
        # --- Step 2: Retrieve documents using a metadata filter to exclude the entire source PPTX ---
        source_pptx_path = slide_doc.metadata.get("source")
        if not source_pptx_path:
            # Fallback if source is not found in metadata
            print(f"Warning: Could not find source metadata in slide {slide_number}. Search may be inaccurate.")
            retriever = get_retriever(mode)
        else:
            metadata_filter = {"source": {"$ne": source_pptx_path}}
            retriever = get_retriever(mode, filter_dict=metadata_filter)

        if not retriever:
            return f"データベースが見つかりませんでした。", None
        retrieved_docs = retriever.invoke(retrieval_query)
        
        # --- Step 3: Generate a response ---
        template = PAGE_SPECIFIC_TEMPLATE_FIRST if is_first_interaction else PAGE_SPECIFIC_TEMPLATE_SUBSEQUENT
        prompt = ChatPromptTemplate.from_template(template)
        response = get_llm_response(llm_question, retrieved_docs, [], prompt, length_instruction, lecture_title)
        
        final_response = f"スライド{slide_number}（タイトル：「{header}」）についてですね。講演ではこんなふうにお話ししましたよ。\n\n---\n\n{response}"
        
        # Add promo periodically
        if interaction_count > 0 and interaction_count % 4 == 0:
            if "Udemyのコースページ" not in final_response:
                final_response += udemy_promo_text
        
        log_interaction(mode, query, retrieved_docs, final_response)
        return final_response, retrieved_docs
    else:
        # For general queries, search the entire DB without filters.
        system_template = CONTEXTUAL_RAG_SYSTEM_TEMPLATE_FIRST if is_first_interaction else CONTEXTUAL_RAG_SYSTEM_TEMPLATE_SUBSEQUENT
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{question}"),
            ]
        )
        response, docs = get_answer(mode, query, history, prompt=prompt, length_instruction=length_instruction, lecture_title=lecture_title)
        
        # Add promo periodically
        if interaction_count > 0 and interaction_count % 4 == 0:
            if "Udemyのコースページ" not in response:
                response += udemy_promo_text
        
        log_interaction(mode, query, docs, response)
        return response, docs
