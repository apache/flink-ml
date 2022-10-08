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

package org.apache.flink.ml.feature.randomsplitter;

import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/** An AlgoOperator which splits a Table into N Tables according to the given weights. */
public class RandomSplitter
        implements AlgoOperator<RandomSplitter>, RandomSplitterParams<RandomSplitter> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public RandomSplitter() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        RowTypeInfo outputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());

        final Double[] weights = getWeights();
        OutputTag<Row>[] outputTags = new OutputTag[weights.length - 1];
        for (int i = 0; i < outputTags.length; ++i) {
            outputTags[i] = new OutputTag<Row>("outputTag_" + i, outputTypeInfo) {};
        }

        SingleOutputStreamOperator<Row> results =
                tEnv.toDataStream(inputs[0])
                        .transform(
                                "SplitterOperator",
                                outputTypeInfo,
                                new SplitterOperator(outputTags, weights));

        Table[] outputTables = new Table[weights.length];
        outputTables[0] = tEnv.fromDataStream(results);

        for (int i = 0; i < outputTags.length; ++i) {
            DataStream<Row> dataStream = results.getSideOutput(outputTags[i]);
            outputTables[i + 1] = tEnv.fromDataStream(dataStream);
        }
        return outputTables;
    }

    private static class SplitterOperator extends AbstractStreamOperator<Row>
            implements OneInputStreamOperator<Row, Row> {
        private final Random random = new Random(0);
        OutputTag<Row>[] outputTag;
        final double[] fractions;

        public SplitterOperator(OutputTag<Row>[] outputTag, Double[] weights) {
            this.outputTag = outputTag;
            this.fractions = new double[weights.length];
            double weightSum = 0.0;
            for (Double weight : weights) {
                weightSum += weight;
            }
            double currentSum = 0.0;
            for (int i = 0; i < fractions.length; ++i) {
                currentSum += weights[i];
                fractions[i] = currentSum / weightSum;
            }
        }

        @Override
        public void processElement(StreamRecord<Row> streamRecord) throws Exception {
            int searchResult = Arrays.binarySearch(fractions, random.nextDouble());
            int index = searchResult < 0 ? -searchResult - 2 : searchResult - 1;
            if (index == -1) {
                output.collect(streamRecord);
            } else {
                output.collect(outputTag[index], streamRecord);
            }
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static RandomSplitter load(StreamTableEnvironment env, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
