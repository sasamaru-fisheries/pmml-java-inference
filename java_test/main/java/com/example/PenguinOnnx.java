package com.example; // パッケージ宣言

import ai.onnxruntime.OnnxMap; // ORTのマップ表現クラス
import ai.onnxruntime.OnnxTensor; // ORTのテンソル表現クラス
import ai.onnxruntime.OrtEnvironment; // ORT環境
import ai.onnxruntime.OrtSession; // ORTセッション
import java.nio.file.Path; // パス表現
import java.nio.file.Paths; // パス生成ユーティリティ
import java.util.Arrays; // 配列の表示などに利用
import java.util.HashMap; // 入力マップ用
import java.util.List; // 可変リスト
import java.util.Map; // マップ

public class PenguinOnnx { // ONNX推論のエントリクラス

    public static void main(String[] args) throws Exception { // 実行エントリポイント
        Path modelPath = Paths.get("..", "model", "penguin.onnx").toAbsolutePath(); // 推論用ONNXモデルのパス
        System.out.println("=== Penguin ONNX inference (Adelie=0, Gentoo=1) ==="); // 見出し
        runModel(modelPath,
                new float[] { 40.3f }, // 数値: bill_length_mm
                new String[] { "Torgersen" }, // カテゴリ: island
                "bill_length_mm + island"); // モデル推論実行
    }

    private static void runModel(Path modelPath, float[] numFeature, String[] catFeature, String label) throws Exception { // モデルを読み込み推論を行う
        try (OrtEnvironment env = OrtEnvironment.getEnvironment(); // ORT環境を取得
             OrtSession.SessionOptions opts = new OrtSession.SessionOptions(); // セッション設定を生成
             OrtSession session = env.createSession(modelPath.toString(), opts)) { // モデルをロードしてセッションを作成

            float[][] billLength = new float[][] { numFeature }; // 数値入力(バッチ1件×1特徴)
            String[][] island = new String[][] { catFeature }; // カテゴリ入力(バッチ1件×1特徴)
            Map<String, OnnxTensor> inputs = new HashMap<>(); // 入力名とテンソルのマップ
            inputs.put("bill_length_mm", OnnxTensor.createTensor(env, billLength)); // 数値テンソルを登録
            inputs.put("island", OnnxTensor.createTensor(env, island)); // カテゴリテンソルを登録

            try (OrtSession.Result result = session.run(inputs)) { // 推論を実行
                String labelOutput = session.getOutputNames().stream() // ラベル出力名を推定
                        .filter(n -> n.toLowerCase().contains("label"))
                        .findFirst()
                        .orElse(session.getOutputNames().iterator().next());

                String probOutput = session.getOutputNames().stream() // 確率出力名を推定
                        .filter(n -> n.toLowerCase().contains("prob"))
                        .findFirst()
                        .orElse(null);

                long predicted = ((long[]) result.get(labelOutput).get().getValue())[0]; // ラベルをlongで取得

                float[] probArray = new float[] {}; // 確率の初期値
                if (probOutput != null) { // 確率出力がある場合のみ処理
                    Object rawProb = result.get(probOutput).get().getValue(); // 確率出力を取得
                    if (rawProb instanceof OnnxMap onnxMap) { // OnnxMapなら中身を取り出す
                        rawProb = onnxMap.getValue();
                    }
                    if (rawProb instanceof float[][] arr && arr.length > 0) { // float配列
                        probArray = arr[0];
                    } else if (rawProb instanceof double[][] arr && arr.length > 0) { // double配列
                        double[] src = arr[0];
                        probArray = new float[src.length];
                        for (int i = 0; i < src.length; i++) probArray[i] = (float) src[i];
                    } else if (rawProb instanceof Map<?, ?> map) { // Map（クラスID -> 確率）
                        probArray = new float[2];
                        Object v0 = map.get(0) != null ? map.get(0) : map.get("0");
                        Object v1 = map.get(1) != null ? map.get(1) : map.get("1");
                        probArray[0] = (v0 instanceof Number n0) ? n0.floatValue() : Float.NaN;
                        probArray[1] = (v1 instanceof Number n1) ? n1.floatValue() : Float.NaN;
                    } else if (rawProb instanceof List<?> list && !list.isEmpty()) { // ListにMapや配列が入る場合
                        Object first = list.get(0);
                        if (first instanceof Map<?, ?> map) {
                            probArray = new float[2];
                            Object v0 = map.get(0) != null ? map.get(0) : map.get("0");
                            Object v1 = map.get(1) != null ? map.get(1) : map.get("1");
                            probArray[0] = (v0 instanceof Number n0) ? n0.floatValue() : Float.NaN;
                            probArray[1] = (v1 instanceof Number n1) ? n1.floatValue() : Float.NaN;
                        } else if (first instanceof float[] fa) {
                            probArray = fa;
                        } else if (first instanceof double[] da) {
                            probArray = new float[da.length];
                            for (int i = 0; i < da.length; i++) probArray[i] = (float) da[i];
                        }
                    }
                }

                System.out.println("[" + label + "]"); // 実行ラベルを表示
                System.out.println("  Model: " + modelPath); // モデルパスを表示
                System.out.println("  Input bill_length_mm: " + Arrays.toString(numFeature)); // 数値入力を表示
                System.out.println("  Input island: " + Arrays.toString(catFeature)); // カテゴリ入力を表示
                System.out.println("  Predicted label: " + predicted); // 予測ラベルを表示
                if (probOutput != null) { // 確率がある場合のみ表示
                    System.out.println("  Probabilities [Adelie, Gentoo]: " + Arrays.toString(probArray)); // 確率を表示
                }
            }
        }
    }

}
