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

package org.apache.flink.ml.classification.logisticregression;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Random;

/**
 * The utility class which provides methods to convert model data from Table to Datastream, and
 * classes to save/load model data.
 */
public class LogisticRegressionModelDataUtil {

    /**
     * Generates a Table containing a {@link LogisticRegressionModelDataSegment} instance with
     * randomly generated coefficient.
     *
     * @param tEnv The environment where to create the table.
     * @param dim The size of generated coefficient.
     * @param seed Random seed.
     */
    public static Table generateRandomModelData(StreamTableEnvironment tEnv, int dim, int seed) {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);
        return tEnv.fromDataStream(
                env.fromElements(1).map(new RandomModelDataGenerator(dim, seed)));
    }

    private static class RandomModelDataGenerator
            implements MapFunction<Integer, LogisticRegressionModelDataSegment> {
        private final int dim;
        private final int seed;

        public RandomModelDataGenerator(int dim, int seed) {
            this.dim = dim;
            this.seed = seed;
        }

        @Override
        public LogisticRegressionModelDataSegment map(Integer integer) throws Exception {
            DenseIntDoubleVector vector = new DenseIntDoubleVector(dim);
            Random random = new Random(seed);
            for (int j = 0; j < dim; j++) {
                vector.values[j] = random.nextDouble();
            }
            return new LogisticRegressionModelDataSegment(vector, 0L);
        }
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<LogisticRegressionModelDataSegment> getModelDataStream(
            Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(
                        x ->
                                new LogisticRegressionModelDataSegment(
                                        x.getFieldAs(0),
                                        x.getFieldAs(1),
                                        x.getFieldAs(2),
                                        x.getFieldAs(3)));
    }

    /**
     * Converts the table model to a data stream of bytes.
     *
     * @param modelDataTable The table of model data.
     * @return The data stream of serialized model data.
     */
    public static DataStream<byte[]> getModelDataByteStream(Table modelDataTable) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();

        return tEnv.toDataStream(modelDataTable)
                .map(
                        x -> {
                            LogisticRegressionModelDataSegment modelData =
                                    new LogisticRegressionModelDataSegment(
                                            x.getFieldAs(0),
                                            x.getFieldAs(1),
                                            x.getFieldAs(2),
                                            x.getFieldAs(3));

                            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
                            modelData.encode(outputStream);
                            return outputStream.toByteArray();
                        });
    }

    /** Data encoder for {@link LogisticRegression} and {@link OnlineLogisticRegression}. */
    public static class ModelDataEncoder implements Encoder<LogisticRegressionModelDataSegment> {

        @Override
        public void encode(LogisticRegressionModelDataSegment modelData, OutputStream outputStream)
                throws IOException {
            modelData.encode(outputStream);
        }
    }

    /** Data decoder for {@link LogisticRegression} and {@link OnlineLogisticRegression}. */
    public static class ModelDataDecoder
            extends SimpleStreamFormat<LogisticRegressionModelDataSegment> {

        @Override
        public Reader<LogisticRegressionModelDataSegment> createReader(
                Configuration configuration, FSDataInputStream inputStream) {
            return new Reader<LogisticRegressionModelDataSegment>() {
                @Override
                public LogisticRegressionModelDataSegment read() throws IOException {
                    try {
                        return LogisticRegressionModelDataSegment.decode(inputStream);
                    } catch (EOFException e) {
                        return null;
                    }
                }

                @Override
                public void close() throws IOException {
                    inputStream.close();
                }
            };
        }

        @Override
        public TypeInformation<LogisticRegressionModelDataSegment> getProducedType() {
            return TypeInformation.of(LogisticRegressionModelDataSegment.class);
        }
    }
}
