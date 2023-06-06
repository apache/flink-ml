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

package org.apache.flink.ml.examples.classification;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegression;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegressionModel;
import org.apache.flink.ml.examples.util.PeriodicSourceFunction;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** Simple program that trains an OnlineLogisticRegression model and uses it for classification. */
public class OnlineLogisticRegressionExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data. Both are infinite streams that periodically
        // sends out provided data to trigger model update and prediction.
        List<Row> trainData1 =
                Arrays.asList(
                        Row.of(Vectors.dense(0.1, 2.), 0.),
                        Row.of(Vectors.dense(0.2, 2.), 0.),
                        Row.of(Vectors.dense(0.3, 2.), 0.),
                        Row.of(Vectors.dense(0.4, 2.), 0.),
                        Row.of(Vectors.dense(0.5, 2.), 0.),
                        Row.of(Vectors.dense(11., 12.), 1.),
                        Row.of(Vectors.dense(12., 11.), 1.),
                        Row.of(Vectors.dense(13., 12.), 1.),
                        Row.of(Vectors.dense(14., 12.), 1.),
                        Row.of(Vectors.dense(15., 12.), 1.));

        List<Row> trainData2 =
                Arrays.asList(
                        Row.of(Vectors.dense(0.2, 3.), 0.),
                        Row.of(Vectors.dense(0.8, 1.), 0.),
                        Row.of(Vectors.dense(0.7, 1.), 0.),
                        Row.of(Vectors.dense(0.6, 2.), 0.),
                        Row.of(Vectors.dense(0.2, 2.), 0.),
                        Row.of(Vectors.dense(14., 17.), 1.),
                        Row.of(Vectors.dense(15., 10.), 1.),
                        Row.of(Vectors.dense(16., 16.), 1.),
                        Row.of(Vectors.dense(17., 10.), 1.),
                        Row.of(Vectors.dense(18., 13.), 1.));

        List<Row> predictData =
                Arrays.asList(
                        Row.of(Vectors.dense(0.8, 2.7), 0.0),
                        Row.of(Vectors.dense(15.5, 11.2), 1.0));

        RowTypeInfo typeInfo =
                new RowTypeInfo(
                        new TypeInformation[] {DenseIntDoubleVectorTypeInfo.INSTANCE, Types.DOUBLE},
                        new String[] {"features", "label"});

        SourceFunction<Row> trainSource =
                new PeriodicSourceFunction(1000, Arrays.asList(trainData1, trainData2));
        DataStream<Row> trainStream = env.addSource(trainSource, typeInfo);
        Table trainTable = tEnv.fromDataStream(trainStream).as("features");

        SourceFunction<Row> predictSource =
                new PeriodicSourceFunction(1000, Collections.singletonList(predictData));
        DataStream<Row> predictStream = env.addSource(predictSource, typeInfo);
        Table predictTable = tEnv.fromDataStream(predictStream).as("features");

        // Creates an online LogisticRegression object and initializes its parameters and initial
        // model data.
        Row initModelData =
                Row.of(Vectors.dense(0.41233679404769874, -0.18088118293232122), 0L, 2L, 0L);
        Table initModelDataTable = tEnv.fromDataStream(env.fromElements(initModelData));
        OnlineLogisticRegression olr =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .setInitialModelData(initModelDataTable);

        // Trains the online LogisticRegression Model.
        OnlineLogisticRegressionModel onlineModel = olr.fit(trainTable);

        // Uses the online LogisticRegression Model for predictions.
        Table outputTable = onlineModel.transform(predictTable)[0];

        // Extracts and displays the results. As training data stream continuously triggers the
        // update of the internal model data, raw prediction results of the same predict dataset
        // would change over time.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector features =
                    (DenseIntDoubleVector) row.getField(olr.getFeaturesCol());
            Double expectedResult = (Double) row.getField(olr.getLabelCol());
            Double predictionResult = (Double) row.getField(olr.getPredictionCol());
            DenseIntDoubleVector rawPredictionResult =
                    (DenseIntDoubleVector) row.getField(olr.getRawPredictionCol());
            System.out.printf(
                    "Features: %-25s \tExpected Result: %s \tPrediction Result: %s \tRaw Prediction Result: %s\n",
                    features, expectedResult, predictionResult, rawPredictionResult);
        }
    }
}
