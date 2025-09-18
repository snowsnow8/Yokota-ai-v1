from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from functions import get_agent_response
from constants import LECTURE_MODES, DB_PATHS

# --- アバター定義 ---
AVATARS = {
    "user": "assets/user_icon_navy.svg",
    "assistant": "assets/yokota_icon.jpg"
}

# --- ページ設定 ---
st.set_page_config(
    page_title="Yokota-AI",
    page_icon="Y",
    layout="wide",
)

# --- データベースの初期化 ---

def initialize_databases():
    """
    Creates or verifies databases. Uses st.session_state to ensure this
    heavy operation only runs once per user session.
    """
    # This function is currently disabled in favor of a local-first approach
    # where databases are pre-built.
    pass

# --- UI要素 ---
st.title("Yokota-AI")
st.caption("講演内容に関する質問に、横田AIがお答えします。")
st.caption("左のサイドバーで講演を選択し、質問を入力してください。")

st.sidebar.header("コンセプト")
st.sidebar.markdown(
    """
    このAIは、私の講演内容をより深く、インタラクティブに学んでいただくための実験的なツールです。講演資料だけでは伝えきれなかったニュアンスや、皆様の疑問に直接お答えします。
    """
)

st.sidebar.header("使い方")

st.sidebar.markdown("**Step 1: 講演を選択**")
selected_mode = st.sidebar.radio(
    "講演選択",  # labelは非表示にするため、ウィジェットの内部的なキーとしてのみ機能
    options=list(LECTURE_MODES.keys()),
    format_func=lambda x: LECTURE_MODES[x],
    label_visibility="collapsed"
)

# --- データベースの初期化 ---
# アプリ起動時にDBの存在を確認する（作成は行わない）
# initialize_databases() # この時点では呼び出さない


# --- チャット履歴の管理 ---

if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0

# 講演モードの変更を監視し、変更されたらチャット履歴をリセット
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = selected_mode

if st.session_state.selected_mode != selected_mode:
    st.session_state.messages = []
    st.session_state.selected_mode = selected_mode
    st.info("講演が切り替わったため、チャット履歴をリセットしました。")

st.sidebar.markdown("**Step 2: 質問を入力**")
st.sidebar.text("メイン画面の入力欄から質問してください。")


st.sidebar.html(
    """
    <style>
        .hint-box {
            font-size: 0.9rem;
            color: var(--gray-80);
            margin-top: 1rem;
            padding-left: 1rem; /* 全体をインデント */
        }
        .hint-box strong {
            color: var(--text-color);
        }
        .hint-box hr {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .hint-box ul {
            padding-left: 1.2rem; /* 箇条書きのインデントを調整 */
            margin-left: 0;
        }
    </style>
    <div class="hint-box">
        <strong>質問のコツ</strong>
        <p>このAIは、私の講演資料をお手元でお持ちの方が、内容をより深く理解するために作られました。そのため、一般的な質問よりも<strong>資料に基づいた具体的な質問</strong>をしていただくと、質の高い回答ができます。</p>
        <hr>
        <strong>具体的な質問の例</strong>
        <ul>
            <li>「今回の講義で、他のブランドの方法論と異なる点はなんでしょうか？」</li>
            <li>「p15の<strong>『〇〇』</strong>について」のように、ページ番号とキーワードを合わせて質問する。</li>
        </ul>
    </div>
    """
)


# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# アプリ再実行時にチャット履歴からメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])

# ユーザーの入力に反応
if prompt := st.chat_input("講演に関する質問をどうぞ"):
    # ユーザーメッセージをチャット履歴に追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # ユーザーメッセージをチャットメッセージコンテナに表示
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    st.session_state.interaction_count += 1

    # アシスタントの応答を取得
    with st.spinner("考え中..."):
        lecture_title = LECTURE_MODES[selected_mode]
        response, source_docs = get_agent_response(
            selected_mode,
            prompt,
            st.session_state.messages,
            lecture_title,
            st.session_state.interaction_count,
        )
    
    # アシスタントの応答をチャットメッセージコンテナに表示
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown(response)
        # if source_docs: # RAGが使用された場合のみ、参照ソースを表示
        #     with st.expander("参照された情報ソース"):
        #         for doc in source_docs:
        #             st.text(f"--- ソース: {doc.metadata.get('source', 'N/A')} ---")
        #             st.caption(doc.page_content)
    
    # アシスタントの応答をチャット履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": response})
