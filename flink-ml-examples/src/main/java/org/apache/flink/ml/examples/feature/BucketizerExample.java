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

import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.feature.bucketizer.Bucketizer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates a Bucketizer instance and uses it for feature engineering. */
public class BucketizerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream = env.fromElements(Row.of(-0.5, 0.0, 1.0, 0.0));
        Table inputTable = tEnv.fromDataStream(inputStream).as("f1", "f2", "f3", "f4");

        // Creates a Bucketizer object and initializes its parameters.
        Double[][] splitsArray =
                new Double[][] {
                    new Double[] {-0.5, 0.0, 0.5},
                    new Double[] {-1.0, 0.0, 2.0},
                    new Double[] {Double.NEGATIVE_INFINITY, 10.0, Double.POSITIVE_INFINITY},
                    new Double[] {Double.NEGATIVE_INFINITY, 1.5, Double.POSITIVE_INFINITY}
                };
        Bucketizer bucketizer =
                new Bucketizer()
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4")
                        .setSplitsArray(splitsArray)
                        .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);

        // Uses the Bucketizer object for feature transformations.
        Table outputTable = bucketizer.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            double[] inputValues = new double[bucketizer.getInputCols().length];
            double[] outputValues = new double[bucketizer.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = (double) row.getField(bucketizer.getInputCols()[i]);
                outputValues[i] = (double) row.getField(bucketizer.getOutputCols()[i]);
            }

            System.out.printf(
                    "Input Values: %s\tOutput Values: %s\n",
                    Arrays.toString(inputValues), Arrays.toString(outputValues));
        }
    }
}
