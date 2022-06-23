/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.examples.evaluation;

import org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluator;
import org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluatorParams;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

/**
 * Simple program that creates a BinaryClassificationEvaluator instance and uses it for evaluation.
 */
public class BinaryClassificationEvaluatorExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(1.0, Vectors.dense(0.1, 0.9)),
                        Row.of(1.0, Vectors.dense(0.2, 0.8)),
                        Row.of(1.0, Vectors.dense(0.3, 0.7)),
                        Row.of(0.0, Vectors.dense(0.25, 0.75)),
                        Row.of(0.0, Vectors.dense(0.4, 0.6)),
                        Row.of(1.0, Vectors.dense(0.35, 0.65)),
                        Row.of(1.0, Vectors.dense(0.45, 0.55)),
                        Row.of(0.0, Vectors.dense(0.6, 0.4)),
                        Row.of(0.0, Vectors.dense(0.7, 0.3)),
                        Row.of(1.0, Vectors.dense(0.65, 0.35)),
                        Row.of(0.0, Vectors.dense(0.8, 0.2)),
                        Row.of(1.0, Vectors.dense(0.9, 0.1)));
        Table inputTable = tEnv.fromDataStream(inputStream).as("label", "rawPrediction");

        // Creates a BinaryClassificationEvaluator object and initializes its parameters.
        BinaryClassificationEvaluator evaluator =
                new BinaryClassificationEvaluator()
                        .setMetricsNames(
                                BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                                BinaryClassificationEvaluatorParams.KS,
                                BinaryClassificationEvaluatorParams.AREA_UNDER_ROC);

        // Uses the BinaryClassificationEvaluator object for evaluations.
        Table outputTable = evaluator.transform(inputTable)[0];

        // Extracts and displays the results.
        Row evaluationResult = outputTable.execute().collect().next();
        System.out.printf(
                "Area under the precision-recall curve: %s\n",
                evaluationResult.getField(BinaryClassificationEvaluatorParams.AREA_UNDER_PR));
        System.out.printf(
                "Area under the receiver operating characteristic curve: %s\n",
                evaluationResult.getField(BinaryClassificationEvaluatorParams.AREA_UNDER_ROC));
        System.out.printf(
                "Kolmogorov-Smirnov value: %s\n",
                evaluationResult.getField(BinaryClassificationEvaluatorParams.KS));
    }
}
