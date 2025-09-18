import os
import shutil
from dotenv import load_dotenv
from functions import load_documents, split_documents, filter_complex_metadata
from constants import DB_PATHS, LECTURE_MODES, EMBEDDING_MODEL

# .envファイルから環境変数を読み込む（特にOPENAI_API_KEY）
load_dotenv()

# LangChainの必要なコンポーネントをインポート
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

def create_all_databases():
    """
    ローカル環境ですべての講演モードのベクトルデータベースを作成または再作成するスクリプト。
    """
    print("Starting database creation process...")

    # OpenAI APIキーが設定されているか確認
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file and add your OpenAI API key to it.")
        return

    # すべての講演モードに対してループ処理
    for mode, title in LECTURE_MODES.items():
        print(f"\n--- Processing mode: '{mode}' ({title}) ---")
        db_path = DB_PATHS.get(mode)

        if not db_path:
            print(f"⚠️ Warning: DB path for mode '{mode}' not defined. Skipping.")
            continue

        # 既存のデータベースがあれば一度削除する
        if os.path.exists(db_path):
            print(f"🗑️ Found existing database at {db_path}. Removing it.")
            try:
                shutil.rmtree(db_path)
                print(f"✅ Successfully removed old database.")
            except OSError as e:
                print(f"❌ Error removing database at {db_path}: {e}")
                continue # 次のモードへ

        # データベースディレクトリを作成
        os.makedirs(db_path, exist_ok=True)

        # 1. ドキュメントの読み込み
        print("📄 Loading documents...")
        documents = load_documents(mode)
        if not documents:
            print(f"❌ FAILURE - No documents were loaded for mode '{mode}'. Check file paths in constants.py.")
            continue
        print(f"👍 Loaded {len(documents)} documents.")

        # 2. ドキュメントの分割
        print("🔪 Splitting documents into chunks...")
        chunks = split_documents(documents)
        if not chunks:
            print(f"❌ FAILURE - Document splitting for mode '{mode}' resulted in 0 chunks.")
            continue
        print(f"👍 Split into {len(chunks)} chunks.")

        # 3. メタデータのフィルタリング
        print("🧹 Filtering complex metadata from chunks...")
        filtered_chunks = filter_complex_metadata(chunks)
        if not filtered_chunks:
            print(f"❌ FAILURE - All chunks were filtered out for mode '{mode}'.")
            continue
        print(f"👍 Filtered chunks remain: {len(filtered_chunks)}.")

        # 4. ベクトルストアの作成と永続化
        print("🧠 Creating and persisting vector store...")
        try:
            embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            
            # ChromaDBのクライアントを初期化
            client = chromadb.PersistentClient(path=db_path)

            # ChromaDBにドキュメントを投入し、永続化する
            Chroma.from_documents(
                documents=filtered_chunks, 
                embedding=embedding_function,
                client=client,
                collection_name=f"collection_{mode}" # 安定版のAPIではコレクション名も指定推奨
            )
            
            print(f"✅ Vector store for '{mode}' created and persisted at {db_path}")

        except Exception as e:
            print(f"❌ An error occurred during vector store creation: {e}")
            import traceback
            print(traceback.format_exc())

    print("\n--- Database creation process finished. ---")

if __name__ == "__main__":
    # このスクリプトが直接実行された場合にのみ、データベース作成処理を呼び出す
    create_all_databases()
