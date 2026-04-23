import pandas as pd

def load_lexicon(path):

    df = pd.read_excel(path)

    entitie_list = df['Name'].astype(str).str.lower().tolist()
    lexique_set = set(entitie_list)

    return entitie_list, lexique_set
