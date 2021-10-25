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

package org.apache.flink.ml.api;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.TestBaseUtils;

import org.apache.commons.collections.IteratorUtils;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Comparator;
import java.util.List;

/** Utility methods for tests. */
public class TestUtils {

    // Executes the given stage using the given inputs and verifies that it produces the expected
    // output.
    public static void executeAndCheckOutput(
            StreamExecutionEnvironment env,
            Stage<?> stage,
            List<List<Integer>> inputs,
            List<Integer> expectedOutput,
            List<List<Integer>> modelDataInputs,
            List<Integer> expectedModelDataOutput)
            throws Exception {
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        Table[] inputTables = new Table[inputs.size()];
        for (int i = 0; i < inputTables.length; i++) {
            inputTables[i] = tEnv.fromDataStream(env.fromCollection(inputs.get(i)));
        }
        Table outputTable = null;
        Table modelDataOutputTable = null;

        if (stage instanceof AlgoOperator) {
            if (modelDataInputs != null) {
                Table[] inputModelDataTables = new Table[modelDataInputs.size()];
                for (int i = 0; i < inputModelDataTables.length; i++) {
                    inputModelDataTables[i] =
                            tEnv.fromDataStream(env.fromCollection(modelDataInputs.get(i)));
                }
                ((Model<?>) stage).setModelData(inputModelDataTables);
            }
            outputTable = ((AlgoOperator<?>) stage).transform(inputTables)[0];
            if (expectedModelDataOutput != null) {
                modelDataOutputTable = ((Model<?>) stage).getModelData()[0];
            }
        } else {
            Estimator<?, ?> estimator = (Estimator<?, ?>) stage;
            Model<?> model = estimator.fit(inputTables);

            if (modelDataInputs != null) {
                Table[] inputModelDataTables = new Table[modelDataInputs.size()];
                for (int i = 0; i < inputModelDataTables.length; i++) {
                    inputModelDataTables[i] =
                            tEnv.fromDataStream(env.fromCollection(modelDataInputs.get(i)));
                }
                model.setModelData(inputModelDataTables);
            }
            outputTable = model.transform(inputTables)[0];
            if (expectedModelDataOutput != null) {
                modelDataOutputTable = model.getModelData()[0];
            }
        }

        List<Integer> output =
                IteratorUtils.toList(
                        tEnv.toDataStream(outputTable, Integer.class).executeAndCollect());
        TestBaseUtils.compareResultCollections(expectedOutput, output, Comparator.naturalOrder());

        if (expectedModelDataOutput != null) {
            List<Integer> modelDataOutput =
                    IteratorUtils.toList(
                            tEnv.toDataStream(modelDataOutputTable, Integer.class)
                                    .executeAndCollect());
            TestBaseUtils.compareResultCollections(
                    expectedModelDataOutput, modelDataOutput, Comparator.naturalOrder());
        }
    }

    /** Encoder for Integer. */
    public static class IntEncoder implements Encoder<Integer> {
        @Override
        public void encode(Integer element, OutputStream stream) throws IOException {
            DataOutputStream dataStream = new DataOutputStream(stream);
            dataStream.writeInt(element);
            dataStream.flush();
        }
    }

    /** Decoder for Integer. */
    public static class IntegerStreamFormat extends SimpleStreamFormat<Integer> {
        @Override
        public Reader<Integer> createReader(Configuration config, FSDataInputStream stream) {
            return new Reader<Integer>() {
                private final DataInputStream dataStream = new DataInputStream(stream);

                @Override
                public Integer read() throws IOException {
                    try {
                        return dataStream.readInt();
                    } catch (EOFException e) {
                        return null;
                    }
                }

                @Override
                public void close() throws IOException {
                    dataStream.close();
                }
            };
        }

        @Override
        public TypeInformation<Integer> getProducedType() {
            return BasicTypeInfo.INT_TYPE_INFO;
        }
    }
}
