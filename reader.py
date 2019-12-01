import pandas as pd
import numpy as np
import os
from io import StringIO

def get_df(path):
    origin_f = open(path, 'r')
    columns = 'start,end,label\n'
    content = columns + origin_f.read().replace(' ', ',')
    df = pd.read_csv(StringIO(content))
    return df

def read_files(path):
    files = os.listdir(path)
    df_dict = {}
    for f in files:
        if not '.lab' in f: continue
        df_dict[f.strip('.lab')] = get_df(os.path.join(path, f))
    return df_dict