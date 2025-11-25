import os

import seaborn as sns  # seabornからペンギンデータセットを取得
from sklearn.compose import ColumnTransformer  # 列ごとの前処理指定に使用
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰モデル
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 標準化とワンホット
from sklearn2pmml import sklearn2pmml  # sklearnモデルをPMML形式に変換
from sklearn2pmml.pipeline import PMMLPipeline  # PMML対応のパイプライン

os.makedirs("model", exist_ok=True)  # モデル保存ディレクトリを作成

# ペンギンデータを読み込み、AdelieとGentooに絞り、欠損行を除去
penguins = sns.load_dataset("penguins")
penguins = penguins[penguins["species"].isin(["Adelie", "Gentoo"])].dropna()

# 種類を0/1ラベルにエンコードし、特徴量と目的変数を用意
penguins["target"] = penguins["species"].map({"Adelie": 0, "Gentoo": 1})
X = penguins[
    ["bill_length_mm", "island"]
]
y = penguins["target"]

# 単一モデルで使用する特徴（数値1+カテゴリ1）
num_features = ["bill_length_mm"]  # 数値
cat_features = ["island"]          # カテゴリ

# 数値は標準化、カテゴリはワンホット化する前処理
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ],
    remainder="drop",
)

# 前処理＋ロジスティック回帰をPMML対応パイプラインとして構築
pipeline = PMMLPipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=200, random_state=42)),
])

pipeline.fit(X[num_features + cat_features], y)  # モデル学習
sklearn2pmml(pipeline, "model/penguin.pmml", with_repr=True)  # PMMLとして保存

print("Saved model/penguin.pmml")  # 保存完了メッセージ
