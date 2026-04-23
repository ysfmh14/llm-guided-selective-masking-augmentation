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
def select_tfidf_mask_positions_no_adjacent(content, mask_prob=0.2):
    tokens = bert_tokenizer.tokenize(content)[:510] 

    # Calculer les scores TF-IDF
    tfidf_scores = dict(zip(feature_names, tfidf_vectorizer.transform([content]).toarray()[0]))

    token_scores = []
    for i, token in enumerate(tokens):
        tok = token.lower()
        if token in ['[CLS]', '[SEP]'] or is_punctuation(tok) or tok in stop_words:
            continue
        
        if tok in lexique_set:
            score = random.uniform(0.0, 0.4)  # Lexique → score faible aléatoire
        else:
            score = tfidf_scores.get(tok, 0.0)  # TF-IDF réel
        token_scores.append((i, score))

    # Trier par score croissant → masquer les tokens les plus faibles en premier
    token_scores.sort(key=lambda x: x[1])

    num_to_mask = max(1, int(len(token_scores) * mask_prob))
    selected_positions = []

    # Sélectionner les positions tout en évitant les adjacents
    for idx, _ in token_scores:
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
