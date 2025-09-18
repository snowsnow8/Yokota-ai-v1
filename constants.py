import os

# Build absolute paths from the directory of this constants.py file.
# The data and db directories are siblings of this file.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Lecture Information ---
LECTURE_MODES = {
    "skill": "AI時代を生き抜く：イノベーション人材の基礎能力５選",
    "brand": "ケーススタディで学ぶ：ブランド戦略のデザイン",
    "design": "バックキャスティングで考える：フューチャーデザイン思考の実践論",
    "mini-design": "速習版：フューチャーデザイン思考のエッセンス",
}

# --- File Paths ---
DATA_DIR = os.path.join(_THIS_DIR, "data")
DB_DIR = os.path.join(_THIS_DIR, "db")

PPTX_PATHS = {
    "skill": os.path.join(DATA_DIR, "⚪︎Udemy_creative-skill_ver_250521_forAI_v3.pptx"),
    "brand": os.path.join(DATA_DIR, "⚪︎Udemy_brand-strategy_to_250524_ver_7_forAI_v2.pptx"),
    "design": os.path.join(DATA_DIR, "⚪︎Udemy_future-design_250527_v3_forAI_v2.pptx"),
    "mini-design": os.path.join(DATA_DIR, "⚪︎Udemy_mini-future-design_250623_v5_forAI_v2.pptx"),
}

PDF_PATHS = {
    "skill": "",
    "brand": "",
    "design": "",
    "mini-design": "",
}

DOCX_PATHS = {
    "skill": [
        os.path.join(DATA_DIR, "innovation_jinzai_complete.docx"),
        os.path.join(DATA_DIR, "innovation_jinzai_summary.docx"),
    ],
    "brand": [
        os.path.join(DATA_DIR, "brand_strategy_detailed_full.docx"),
        os.path.join(DATA_DIR, "brand_strategy_summary_with_tips.docx"),
    ],
    "design": [
        os.path.join(DATA_DIR, "future_design_lecture_transcript_corrected.docx"),
        os.path.join(DATA_DIR, "future_design_summary.docx"),
    ],
    "mini-design": [
        os.path.join(DATA_DIR, "future_design_lecture_transcript_corrected.docx"),
        os.path.join(DATA_DIR, "future_design_summary.docx"),
    ],
}

DB_PATHS = {
    "skill": os.path.join(DB_DIR, "skill_db"),
    "brand": os.path.join(DB_DIR, "brand_db"),
    "design": os.path.join(DB_DIR, "design_db"),
    "mini-design": os.path.join(DB_DIR, "mini_design_db"),
}

FAQ_PATH = os.path.join(DATA_DIR, "FAQ_Questions_List_v1.csv")

# --- Model Information ---
# Embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"

# LLM for generation
GENERATION_MODEL = "gpt-4.1-mini"
