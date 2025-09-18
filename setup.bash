#!/bin/bash
set -e # コマンドがエラーになったら、ただちにスクリプトを終了する

# --- より堅牢な環境構築スクリプト ---

# スクリプトのあるディレクトリに移動
dirname=$(dirname "$0")
cd "$dirname"

# 1. 仮想環境をクリーンに再作成
echo "古い仮想環境 .venv を削除し、再作成します..."
rm -rf .venv
python3 -m venv .venv
echo "仮想環境を .venv に作成しました。"

# 2. 仮想環境内のpipを直接使って、pip自体をアップグレード
echo "仮想環境内のpipをアップグレードします..."
./.venv/bin/pip install --upgrade pip

# 3. requirements.txtからライブラリをインストール
if [ -f requirements.txt ]; then
  echo "requirements.txt からライブラリをインストールします..."
  ./.venv/bin/pip install -r requirements.txt
else
  echo "警告: requirements.txt が見つかりません。"
fi

# 4. NLTKのデータパッケージをダウンロード
# unstructuredライブラリが内部で使用するNLTKのデータを事前にダウンロードし、
# アプリ実行時のネットワークアクセスを回避する
echo "NLTKデータパッケージ（'punkt', 'averaged_perceptron_tagger'）をダウンロードします..."
./.venv/bin/python -m nltk.downloader punkt
./.venv/bin/python -m nltk.downloader averaged_perceptron_tagger
echo "NLTKデータパッケージのダウンロードが完了しました。"

# 5. 最終診断
echo ""
echo "--- インストール完了後の最終診断 ---"
echo "Pythonのバージョン: $(./.venv/bin/python --version)"
echo "rank_bm25のインストール状況を確認します..."

# grep -q でrank_bm25が見つかるかチェック
if ./.venv/bin/pip list | grep -q 'rank-bm25'; then
    echo "✅ 成功: rank_bm25 は正しくインストールされています。"
    ./.venv/bin/pip list | grep 'rank-bm25'
else
    echo "❌ 失敗: rank_bm25 のインストールに失敗しました。"
    echo "pip install のログにエラーがないか確認してください。"
    exit 1
fi
echo "----------------------------------"
echo "セットアップが正常に完了しました。"