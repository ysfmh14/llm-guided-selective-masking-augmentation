from transformers import BertTokenizer, BertForMaskedLM

def load_bert(entitie_list):

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(entitie_list)

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    return tokenizer, model
