from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

def load_llm(model_name, hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    return tokenizer, model


def llm_scoring(tokens, mask_idx, candidates, bert_tokenizer, llm_model, llm_tokenizer):

    sentences = []
    for tok in candidates:
        temp = tokens.copy()
        temp[mask_idx] = tok
        sentences.append(bert_tokenizer.convert_tokens_to_string(temp))

    inputs = llm_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model(**inputs)

    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["input_ids"][:, 1:]
    shift_mask = inputs["attention_mask"][:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

    scores = (gathered * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)

    best_idx = scores.argmax().item()
    return candidates[best_idx]
