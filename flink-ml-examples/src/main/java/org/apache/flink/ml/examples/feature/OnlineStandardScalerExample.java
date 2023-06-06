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

import org.apache.flink.api.common.eventtime.SerializableTimestampAssigner;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.time.Time;
import org.apache.flink.ml.common.window.EventTimeTumblingWindows;
import org.apache.flink.ml.feature.standardscaler.OnlineStandardScaler;
import org.apache.flink.ml.feature.standardscaler.OnlineStandardScalerModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.List;

/** Simple program that trains a OnlineStandardScaler model and uses it for feature engineering. */
public class OnlineStandardScalerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        List<Row> inputData =
                Arrays.asList(
                        Row.of(0L, Vectors.dense(-2.5, 9, 1)),
                        Row.of(1000L, Vectors.dense(1.4, -5, 1)),
                        Row.of(2000L, Vectors.dense(2, -1, -2)),
                        Row.of(6000L, Vectors.dense(0.7, 3, 1)),
                        Row.of(7000L, Vectors.dense(0, 1, 1)),
                        Row.of(8000L, Vectors.dense(0.5, 0, -2)),
                        Row.of(9000L, Vectors.dense(0.4, 1, 1)),
                        Row.of(10000L, Vectors.dense(0.3, 2, 1)),
                        Row.of(11000L, Vectors.dense(0.5, 1, -2)));

        DataStream<Row> inputStream = env.fromCollection(inputData);

        DataStream<Row> inputStreamWithEventTime =
                inputStream.assignTimestampsAndWatermarks(
                        WatermarkStrategy.<Row>forMonotonousTimestamps()
                                .withTimestampAssigner(
                                        (SerializableTimestampAssigner<Row>)
                                                (element, recordTimestamp) ->
                                                        element.getFieldAs(0)));

        Table inputTable =
                tEnv.fromDataStream(
                                inputStreamWithEventTime,
                                Schema.newBuilder()
                                        .column("f0", DataTypes.BIGINT())
                                        .column(
                                                "f1",
                                                DataTypes.RAW(
                                                        DenseIntDoubleVectorTypeInfo.INSTANCE))
                                        .columnByMetadata("rowtime", "TIMESTAMP_LTZ(3)")
                                        .watermark("rowtime", "SOURCE_WATERMARK()")
                                        .build())
                        .as("id", "input");

        // Creates an OnlineStandardScaler object and initializes its parameters.
        long windowSizeMs = 3000;
        OnlineStandardScaler onlineStandardScaler =
                new OnlineStandardScaler()
                        .setWindows(EventTimeTumblingWindows.of(Time.milliseconds(windowSizeMs)));

        // Trains the OnlineStandardScaler Model.
        OnlineStandardScalerModel model = onlineStandardScaler.fit(inputTable);

        // Uses the OnlineStandardScaler Model for predictions.
        Table outputTable = model.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector inputValue =
                    (DenseIntDoubleVector) row.getField(onlineStandardScaler.getInputCol());
            DenseIntDoubleVector outputValue =
                    (DenseIntDoubleVector) row.getField(onlineStandardScaler.getOutputCol());
            long modelVersion = row.getFieldAs(onlineStandardScaler.getModelVersionCol());
            System.out.printf(
                    "Input Value: %s\tOutput Value: %-65s\tModel Version: %s\n",
                    inputValue, outputValue, modelVersion);
        }
    }
}
