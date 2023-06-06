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

import org.apache.flink.ml.feature.countvectorizer.CountVectorizer;
import org.apache.flink.ml.feature.countvectorizer.CountVectorizerModel;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/**
 * Simple program that trains a {@link CountVectorizer} model and uses it for feature engineering.
 */
public class CountVectorizerExample {

    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> dataStream =
                env.fromElements(
                        Row.of((Object) new String[] {"a", "c", "b", "c"}),
                        Row.of((Object) new String[] {"c", "d", "e"}),
                        Row.of((Object) new String[] {"a", "b", "c"}),
                        Row.of((Object) new String[] {"e", "f"}),
                        Row.of((Object) new String[] {"a", "c", "a"}));
        Table inputTable = tEnv.fromDataStream(dataStream).as("input");

        // Creates an CountVectorizer object and initialize its parameters
        CountVectorizer countVectorizer = new CountVectorizer();

        // Trains the CountVectorizer model
        CountVectorizerModel model = countVectorizer.fit(inputTable);

        // Uses the CountVectorizer model for predictions.
        Table outputTable = model.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            String[] inputValue = (String[]) row.getField(countVectorizer.getInputCol());
            SparseIntDoubleVector outputValue =
                    (SparseIntDoubleVector) row.getField(countVectorizer.getOutputCol());
            System.out.printf(
                    "Input Value: %-15s \tOutput Value: %s\n",
                    Arrays.toString(inputValue), outputValue.toString());
        }
    }
}
