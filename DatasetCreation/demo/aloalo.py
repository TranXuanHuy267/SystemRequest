from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("model_text2table_5")
model_matching = SentenceTransformer("model_matching")
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_params_2 = sum(p.numel() for p in model_matching.parameters())
print(pytorch_total_params)
print(pytorch_total_params_2)