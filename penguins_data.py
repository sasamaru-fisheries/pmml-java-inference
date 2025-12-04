import os

import seaborn as sns  
import pandas as pd

os.makedirs("model", exist_ok=True)  # モデル保存用ディレクトリを作成

# ペンギンデータを読み込み、AdelieとGentooの2値分類データに絞る
penguins = sns.load_dataset("penguins")
print(penguins)
print(penguins.isnull().sum())

