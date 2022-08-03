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

import org.apache.flink.ml.feature.elementwiseproduct.ElementwiseProduct;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/**
 * Simple program that creates a ElementwiseProduct instance and uses it for feature engineering.
 */
public class ElementwiseProductExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(0, Vectors.dense(1.1, 3.2)), Row.of(1, Vectors.dense(2.1, 3.1)));

        Table inputTable = tEnv.fromDataStream(inputStream).as("id", "vec");

        // Creates an ElementwiseProduct object and initializes its parameters.
        ElementwiseProduct elementwiseProduct =
                new ElementwiseProduct()
                        .setInputCol("vec")
                        .setOutputCol("outputVec")
                        .setScalingVec(Vectors.dense(1.1, 1.1));

        // Transforms input data.
        Table outputTable = elementwiseProduct.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            Vector inputValue = (Vector) row.getField(elementwiseProduct.getInputCol());
            Vector outputValue = (Vector) row.getField(elementwiseProduct.getOutputCol());
            System.out.printf("Input Value: %s \tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}
