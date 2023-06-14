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

package org.apache.flink.ml.regression.linearregression;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorSerializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Model data of {@link LinearRegressionModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
public class LinearRegressionModelData {

    public DenseIntDoubleVector coefficient;

    public LinearRegressionModelData(DenseIntDoubleVector coefficient) {
        this.coefficient = coefficient;
    }

    public LinearRegressionModelData() {}

    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<LinearRegressionModelData> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(x -> new LinearRegressionModelData((DenseIntDoubleVector) x.getField(0)));
    }

    /** Data encoder for {@link LinearRegressionModel}. */
    public static class ModelDataEncoder implements Encoder<LinearRegressionModelData> {
        private final DenseIntDoubleVectorSerializer serializer =
                new DenseIntDoubleVectorSerializer();

        @Override
        public void encode(LinearRegressionModelData modelData, OutputStream outputStream)
                throws IOException {
            serializer.serialize(
                    modelData.coefficient, new DataOutputViewStreamWrapper(outputStream));
        }
    }

    /** Data decoder for {@link LinearRegressionModel}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<LinearRegressionModelData> {

        @Override
        public Reader<LinearRegressionModelData> createReader(
                Configuration configuration, FSDataInputStream inputStream) {
            return new Reader<LinearRegressionModelData>() {
                private final DenseIntDoubleVectorSerializer serializer =
                        new DenseIntDoubleVectorSerializer();

                @Override
                public LinearRegressionModelData read() throws IOException {
                    try {
                        DenseIntDoubleVector coefficient =
                                serializer.deserialize(new DataInputViewStreamWrapper(inputStream));
                        return new LinearRegressionModelData(coefficient);
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
        public TypeInformation<LinearRegressionModelData> getProducedType() {
            return TypeInformation.of(LinearRegressionModelData.class);
        }
    }
}
