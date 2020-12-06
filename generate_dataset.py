import pandas as pd
import numpy as np

data = pd.read_csv('data/ToLD-BR_alpha.csv')

data["toxic"] = 0
data["toxic_1"] = 0
data["toxic_2"] = 0
data["toxic_3"] = 0
data["homophobia"] = 0
data["obscene"] = 0
data["insult"] = 0
data["racism"] = 0
data["misogyny"] = 0
data["xenophobia"] = 0
for i, row in data.iterrows():
    count = 0
    if 1 in list(row[["homophobia_1", "obscene_1", "insult_1", "racism_1", "misogyny_1", "xenophobia_1"]]):
        data.loc[i, "toxic_1"] = 1
        count += 1
    if 1 in list(row[["homophobia_2", "obscene_2", "insult_2", "racism_2", "misogyny_2", "xenophobia_2"]]):
        data.loc[i, "toxic_2"] = 1
        count += 1
    if 1 in list(row[["homophobia_3", "obscene_3", "insult_3", "racism_3", "misogyny_3", "xenophobia_3"]]):
        data.loc[i, "toxic_3"] = 1
        count += 1

    data.loc[i, "homophobia"] = np.sum(list(row[["homophobia_1", "homophobia_2", "homophobia_3"]]))

    data.loc[i, "obscene"] = np.sum(list(row[["obscene_1", "obscene_2", "obscene_3"]]))

    data.loc[i, "insult"] = np.sum(list(row[["insult_1", "insult_2", "insult_3"]]))

    data.loc[i, "racism"] = np.sum(list(row[["racism_1", "racism_2", "racism_3"]]))

    data.loc[i, "misogyny"] = np.sum(list(row[["misogyny_1", "misogyny_2", "misogyny_3"]]))

    data.loc[i, "xenophobia"] = np.sum(list(row[["xenophobia_1", "xenophobia_2", "xenophobia_3"]]))

    if count >= 1:
        data.loc[i, "toxic"] = 1

data[["text", "toxic", "homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia"]].to_csv('data/ToLD-BR.csv')
