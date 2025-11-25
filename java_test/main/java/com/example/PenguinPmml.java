package com.example; // パッケージ宣言

import java.nio.file.Files; // ファイル入出力
import java.nio.file.Path; // パス表現
import java.nio.file.Paths; // パス生成ユーティリティ
import java.util.LinkedHashMap; // 順序付きマップ
import java.util.Map; // マップ

import org.dmg.pmml.PMML; // PMMLモデル表現
import org.jpmml.evaluator.Evaluator; // 評価器インターフェース
import org.jpmml.evaluator.InputField; // 入力フィールド情報
import org.jpmml.evaluator.ModelEvaluatorBuilder; // 評価器ビルダー
import org.jpmml.evaluator.ProbabilityDistribution; // 確率分布
import org.jpmml.evaluator.TargetField; // 予測ターゲット情報
import org.jpmml.model.PMMLUtil; // PMMLユーティリティ

public class PenguinPmml { // PMML推論のエントリクラス

    public static void main(String[] args) throws Exception { // 実行エントリポイント
        Path modelPath = Paths.get("..", "model", "penguin.pmml").toAbsolutePath(); // PMMLモデルのパス

        System.out.println("=== Penguin PMML inference (Adelie=0, Gentoo=1) ==="); // 見出し
        runModel(modelPath, Map.<String, Object>of( // 推論を実行
                "bill_length_mm", 40.3, // 嘴長
                "island", "Torgersen"   // 島（カテゴリ）
        ), "bill_length_mm + island"); // ラベル文字列
    }

    private static void runModel(Path modelPath, Map<String, Object> rawFeatures, String label) throws Exception { // モデルを読み込み推論を実施
        try (var is = Files.newInputStream(modelPath)) { // PMMLファイルをストリームで開く
            PMML pmml = PMMLUtil.unmarshal(is); // PMMLをパース
            Evaluator evaluator = new ModelEvaluatorBuilder(pmml).build(); // 評価器を生成
            evaluator.verify(); // モデルを検証

            Map<String, Object> arguments = new LinkedHashMap<>(); // 入力マップを生成
            for (InputField inputField : evaluator.getInputFields()) { // すべての入力フィールドを処理
                String name = inputField.getName(); // フィールド名
                Object rawValue = rawFeatures.get(name); // 元データから値を取得
                arguments.put(name, inputField.prepare(rawValue)); // 必要な前処理を適用してマップに入れる
            }

            Map<String, ?> results = evaluator.evaluate(arguments); // 推論を実行

            TargetField target = evaluator.getTargetFields().get(0); // 予測ターゲット情報
            ProbabilityDistribution<?> dist = (ProbabilityDistribution<?>) results.get(target.getName()); // 分布を取得
            Object predicted = dist.getResult(); // ラベルを取得

            double probAdelie = dist.getProbability(0); // Adelieの確率
            double probGentoo = dist.getProbability(1); // Gentooの確率

            System.out.println("[" + label + "]"); // ラベル文字列表示
            System.out.println("  Model: " + modelPath); // モデルパス表示
            System.out.println("  Input: " + rawFeatures); // 入力値表示
            System.out.println("  Predicted label: " + predicted); // 予測ラベル表示
            System.out.println("  Probabilities {Adelie=0, Gentoo=1}: [" // 確率表示
                    + probAdelie + ", " + probGentoo + "]"); // Adelie/Gentooの確率
        }
    }

}
