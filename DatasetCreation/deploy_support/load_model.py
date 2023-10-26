from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_matching = SentenceTransformer("Huy1432884/function_retrieval")
model_matching.save("../deploy/model_matching")

model = AutoModelForSeq2SeqLM.from_pretrained(
    "Huy1432884/db_retrieval",
    use_auth_token="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"
)
model.save_pretrained("../deploy/model")

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt", src_lang="vi_VN", tgt_lang="vi_VN"
)

tokenizer.save_pretrained("../deploy/tokenizer")


