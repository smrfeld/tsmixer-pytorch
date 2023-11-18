import pandas as pd

def load_etdataset(csv_file: str):

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file, parse_dates=['date'])

    # Display the DataFrame
    print(df)
