from common.llm import load_llm, llm_scoring
from common.bert import load_bert
from common.data import load_dataset
from common.lexicon import load_lexicon
from config import *

import torch
from tqdm import tqdm
import pandas as pd

# Load
entitie_list, lexique_set = load_lexicon(LEXICON_PATH)
bert_tokenizer, bert_model = load_bert(entitie_list)
llm_tokenizer, llm_model = load_llm(LLM_MODEL, HF_TOKEN)

df = load_dataset(DATA_PATH, TEXT_COL, LABEL_COL)

# Strategy spécifique
def select_mask_positions(content, mask_prob=0.2):
    tokens = bert_tokenizer.tokenize(content)

    candidate_indices = [
        i for i, tok in enumerate(tokens)
        if not is_punctuation(tok)
        and tok.lower() not in stop_words
        and tok.lower() not in lexique_set
    ]

    num_to_mask = max(1, int(len(candidate_indices) * mask_prob))
    random.shuffle(candidate_indices)

    selected = []
    for idx in candidate_indices:
        if all(abs(idx - s) > 1 for s in selected):
            selected.append(idx)
        if len(selected) >= num_to_mask:
            break

    return selected, tokens

# Augmentation
def augment(text):

    tokens = bert_tokenizer.tokenize(text)
    masked_positions = select_mask_positions(tokens)

    input_tokens = tokens.copy()

    for i in masked_positions:
        input_tokens[i] = "[MASK]"

    input_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + input_tokens + ["[SEP]"])
    input_tensor = torch.tensor([input_ids])

    with torch.no_grad():
        predictions = bert_model(input_tensor).logits

    for idx in masked_positions:
        pred_idx = idx + 1

        top_ids = torch.topk(predictions[0, pred_idx], TOP_K).indices.tolist()
        candidates = bert_tokenizer.convert_ids_to_tokens(top_ids)

        best = llm_scoring(input_tokens, idx, candidates,
                           bert_tokenizer, llm_model, llm_tokenizer)

        input_tokens[idx] = best

    return bert_tokenizer.convert_tokens_to_string(input_tokens)

# Run
augmented = []

for text in tqdm(df[TEXT_COL]):
    augmented.append(augment(text))

df["augmented"] = augmented
df.to_excel("output_AuSeMa-Nl-LLM.xlsx", index=False)
