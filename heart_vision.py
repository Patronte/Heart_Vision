import pandas as pd
import matplotlib.pyplot as plt




df_X = pd.read_csv("Dataset_storico/HEARTVISION_dataset_completo.csv")

df_y = pd.read_csv("Dataset_questionari/Framingham_form.csv")

df_y = df_y.drop(columns=["Timestamp", "Username"])
print(df_y)
