import csv
import pandas as pd
from pandas.core.frame import DataFrame

def split_datetime(df: DataFrame, col_to_split: str, cols: list, ) -> DataFrame:

    df[cols[0]] = pd.to_datetime(df[col_to_split], dayfirst = True).dt.date
    df[cols[1]] = pd.to_datetime(df[col_to_split]).dt.time

    return df
    

def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


