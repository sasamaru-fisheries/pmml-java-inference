"""Load a PMML model and run sample predictions in Python."""

import numpy as np
import pandas as pd
from pypmml import Model

# PMMLモデルを読み込む
model = Model.load("model/penguin.pmml")

# サンプル入力（欠損も含む）
data = pd.DataFrame([
    {"bill_length_mm": np.nan, "island": ""},            # 欠損ケース
    {"bill_length_mm": 40.3, "island": "Torgersen"},     # 値ありケース
])

# 推論実行
pred = model.predict(data)

print("Input:")
print(data)
print("\nPrediction:")
print(pred)
