import pandas as pd

def load_structured_data(path):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    print("Loader ready")
