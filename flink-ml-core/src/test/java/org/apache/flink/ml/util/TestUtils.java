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

package org.apache.flink.ml.util;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.api.Stage;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.SparseVectorTypeInfo;
import org.apache.flink.ml.servable.api.TransformerServable;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.types.DataType;
import org.apache.flink.test.util.TestBaseUtils;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.ArrayUtils;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Method;
import java.util.Comparator;
import java.util.List;

/** Utility methods for tests. */
public class TestUtils {

    /**
     * Gets a {@link StreamExecutionEnvironment} with the most common configurations of the Flink ML
     * program as well as the given extra configuration.
     */
    public static StreamExecutionEnvironment getExecutionEnvironment(Configuration extraConfig) {
        StreamExecutionEnvironment env = getExecutionEnvironment();
        env.configure(extraConfig);
        return env;
    }

    /**
     * Gets a {@link StreamExecutionEnvironment} with the most common configurations of the Flink ML
     * program.
     */
    public static StreamExecutionEnvironment getExecutionEnvironment() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.getConfig().disableGenericTypes();
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        return env;
    }

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

    /**
     * Saves a stage to filesystem and reloads it by invoking the static load() method of the given
     * stage.
     */
    public static <T extends Stage<T>> T saveAndReload(
            StreamTableEnvironment tEnv, T stage, String path) throws Exception {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);

        stage.save(path);
        try {
            env.execute();
        } catch (RuntimeException e) {
            if (!e.getMessage()
                    .equals("No operators defined in streaming topology. Cannot execute.")) {
                throw e;
            }
        }

        Method method =
                stage.getClass().getMethod("load", StreamTableEnvironment.class, String.class);
        return (T) method.invoke(null, tEnv, path);
    }

    /**
     * Saves a transformer to filesystem and reloads the matadata as a servable by invoking the
     * static loadServable() method.
     */
    public static <T extends TransformerServable<T>> T saveAndLoadServable(
            StreamTableEnvironment tEnv, Transformer transformer, String path) throws Exception {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);

        transformer.save(path);
        try {
            env.execute();
        } catch (RuntimeException e) {
            if (!e.getMessage()
                    .equals("No operators defined in streaming topology. Cannot execute.")) {
                throw e;
            }
        }

        Method method = transformer.getClass().getMethod("loadServable", String.class);
        return (T) method.invoke(null, path);
    }

    /**
     * Converts data types in the table to sparse types and integer types.
     *
     * <ul>
     *   <li>If a column in the table is of DenseVector type, converts it to SparseVector.
     *   <li>If a column in the table is of Double type, converts it to integer.
     * </ul>
     */
    public static Table convertDataTypesToSparseInt(StreamTableEnvironment tEnv, Table table) {
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(table.getResolvedSchema());
        TypeInformation<?>[] fieldTypes = inputTypeInfo.getFieldTypes();
        for (int i = 0; i < fieldTypes.length; i++) {
            if (fieldTypes[i].getTypeClass().equals(DenseVector.class)) {
                fieldTypes[i] = SparseVectorTypeInfo.INSTANCE;
            } else if (fieldTypes[i].getTypeClass().equals(Double.class)) {
                fieldTypes[i] = Types.INT;
            }
        }

        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(fieldTypes),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames()));
        DataStream<Row> dataStream = tEnv.toDataStream(table);
        dataStream =
                dataStream.map(
                        new MapFunction<Row, Row>() {
                            @Override
                            public Row map(Row row) {
                                int arity = row.getArity();
                                for (int i = 0; i < arity; i++) {
                                    Object obj = row.getField(i);
                                    if (obj instanceof Vector) {
                                        row.setField(i, ((Vector) obj).toSparse());
                                    } else if (obj instanceof Number) {
                                        row.setField(i, ((Number) obj).intValue());
                                    }
                                }
                                return row;
                            }
                        },
                        outputTypeInfo);
        return tEnv.fromDataStream(dataStream);
    }

    /** Gets the types of data in each column of the input table. */
    public static Class<?>[] getColumnDataTypes(Table table) {
        return table.getResolvedSchema().getColumnDataTypes().stream()
                .map(DataType::getConversionClass)
                .toArray(Class<?>[]::new);
    }

    /** Note: this comparator imposes orderings that are inconsistent with equals. */
    public static int compare(Vector first, Vector second) {
        if (first.size() != second.size()) {
            return Integer.compare(first.size(), second.size());
        } else {
            for (int i = 0; i < first.size(); i++) {
                int cmp = Double.compare(first.get(i), second.get(i));
                if (cmp != 0) {
                    return cmp;
                }
            }
        }
        return 0;
    }
}
