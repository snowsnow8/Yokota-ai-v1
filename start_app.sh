#!/bin/bash
# Streamlitアプリケーションを起動するスクリプト

# スクリプトのディレクトリに移動
dirname=$(dirname "$0")
cd "$dirname"

# 仮想環境の有効化
# このスクリプトを実行する前に、必ず `bash setup.bash` を実行して環境を構築しておいてください。
if [ ! -d ".venv" ]; then
    echo "エラー: 仮想環境 .venv が見つかりません。"
    echo "最初に `bash setup.bash` を実行して環境をセットアップしてください。"
    exit 1
fi
source .venv/bin/activate

# .envファイルが存在すれば読み込む
if [ -f .env ]; then
  echo ".envファイルが見つかりました。環境変数を読み込みます。"
  set -a
  source .env
  set +a
fi

# Streamlitアプリの起動
# `streamlit`コマンドを仮想環境内のフルパスで直接指定することで、
# グローバル環境ではなく、確実に正しい環境で実行する。
echo "アプリケーションを起動します..."
./.venv/bin/streamlit run main.py