import pandas as pd

def load_dataset(path, text_col, label_col):
    df = pd.read_excel(path)
    assert text_col in df.columns and label_col in df.columns
    return df
