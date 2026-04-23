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
    tokens = bert_tokenizer.tokenize(content)[:510]


    # --- Séparer en deux groupes ---
    lexique_indices = [
        i for i, token in enumerate(tokens)
        if token not in ['[CLS]', '[SEP]']
        and not is_punctuation(token)
        and token.lower() in lexique_set  # mots DU lexique
    ]

    other_indices = [
        i for i, token in enumerate(tokens)
        if token not in ['[CLS]', '[SEP]']
        and not is_punctuation(token)
        and token.lower() not in stop_words
        and token.lower() not in lexique_set  # mots hors lexique
    ]

    num_to_mask = max(1, int((len(lexique_indices) + len(other_indices)) * mask_prob))
    selected_positions = []

    # --- Étape 1 : masquer en priorité les mots du lexique ---
    random.shuffle(lexique_indices)
    for idx in lexique_indices:
        if all(abs(idx - sel) > 1 for sel in selected_positions):
            selected_positions.append(idx)
        if len(selected_positions) >= num_to_mask:
            break

    # --- Étape 2 : compléter avec les autres mots si quota non atteint ---
    remaining = num_to_mask - len(selected_positions)
    if remaining > 0:
        random.shuffle(other_indices)
        for idx in other_indices:
            if all(abs(idx - sel) > 1 for sel in selected_positions):
                selected_positions.append(idx)
            if len(selected_positions) >= num_to_mask:
                break

    return selected_positions, tokens

# Augmentation
def augment(text):

    masked_positions, tokens= select_mask_positions(tokens)

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
