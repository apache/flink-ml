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

package org.apache.flink.ml.benchmark.classification;

import org.apache.flink.ml.api.Stage;
import org.apache.flink.ml.benchmark.BenchmarkContext;
import org.apache.flink.ml.benchmark.BenchmarkStage;
import org.apache.flink.ml.benchmark.Constants;
import org.apache.flink.ml.benchmark.Utils;
import org.apache.flink.ml.benchmark.data.DataGenerator;
import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModel;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelData;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

import java.util.Arrays;
import java.util.Random;

/** Benchmark test for {@link LogisticRegression}. */
public class LogisticRegressionBenchmark implements BenchmarkStage<LogisticRegression> {

    @Override
    public Table[] getTrainData(BenchmarkContext context) {
        DataStream<Vector> data =
                DataGenerator.generateContinuousFeatures(
                        context.env,
                        context.params.numExamples,
                        context.params.randomSeed,
                        context.params.numPartitions,
                        context.params.numFeatures);
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(context.env);
        Table trainTable = tEnv.fromDataStream(data).as(Constants.FEATURE_COL);

        int modelDim = context.params.numFeatures;
        double[] modelDataArray = new double[modelDim];
        Random random = new Random(context.params.randomSeed);
        Arrays.fill(modelDataArray, random.nextDouble());
        DataStream<LogisticRegressionModelData> modelDataDataStream =
                context.env.fromElements(
                        new LogisticRegressionModelData(Vectors.dense(modelDataArray)));
        Table modelData = tEnv.fromDataStream(modelDataDataStream);

        LogisticRegressionModel model =
                new LogisticRegressionModel()
                        .setModelData(modelData)
                        .setFeaturesCol(Constants.FEATURE_COL)
                        .setPredictionCol(Constants.PREDICT_COL);
        Table[] trainTables =
                Utils.getTrainDataFromTrueModelAndData(model, new Table[] {trainTable});

        return trainTables;
    }

    @Override
    public Stage<LogisticRegression> getStage(BenchmarkContext context) {
        return new LogisticRegression()
                .setFeaturesCol(Constants.FEATURE_COL)
                .setPredictionCol(Constants.PREDICT_COL)
                .setLabelCol(Constants.LABEL_COL)
                .setGlobalBatchSize(context.params.globalBatchSize)
                .setLearningRate(context.params.learningRate)
                .setReg(context.params.reg);
    }
}
