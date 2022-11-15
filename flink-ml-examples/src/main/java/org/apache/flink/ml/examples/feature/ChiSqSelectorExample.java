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

package org.apache.flink.ml.examples.feature;

import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.feature.chisqselector.ChiSqSelector;
import org.apache.flink.ml.feature.chisqselector.ChiSqSelectorModel;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

import static org.apache.flink.ml.feature.chisqselector.ChiSqSelectorParams.NUM_TOP_FEATURES_TYPE;

/** Simple program that trains a ChiSqSelector model and uses it for feature engineering. */
public class ChiSqSelectorExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromCollection(
                        Arrays.asList(
                                Row.of(
                                        0.0,
                                        Vectors.sparse(
                                                6,
                                                new int[] {0, 1, 3, 4},
                                                new double[] {6.0, 7.0, 7.0, 6.0})),
                                Row.of(
                                        1.0,
                                        Vectors.sparse(
                                                6,
                                                new int[] {1, 2, 4, 5},
                                                new double[] {9.0, 6.0, 5.0, 9.0})),
                                Row.of(
                                        1.0,
                                        Vectors.sparse(
                                                6,
                                                new int[] {1, 2, 4, 5},
                                                new double[] {9.0, 3.0, 5.0, 5.0})),
                                Row.of(1.0, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0)),
                                Row.of(2.0, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0)),
                                Row.of(2.0, Vectors.dense(8.0, 9.0, 6.0, 4.0, 0.0, 0.0))),
                        new RowTypeInfo(Types.DOUBLE, VectorTypeInfo.INSTANCE));
        Table inputTable = tEnv.fromDataStream(inputStream).as("label", "features");

        // Creates a ChiSqSelector object and initializes its parameters.
        ChiSqSelector selector =
                new ChiSqSelector()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setOutputCol("prediction")
                        .setSelectorType(NUM_TOP_FEATURES_TYPE)
                        .setNumTopFeatures(1);

        // Trains the ChiSqSelector Model.
        ChiSqSelectorModel model = selector.fit(inputTable);

        // Uses the ChiSqSelector Model for predictions.
        Table outputTable = model.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            Vector inputValue = row.getFieldAs("features");
            Vector outputValue = row.getFieldAs("prediction");
            System.out.printf("Input Value: %s \tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}
