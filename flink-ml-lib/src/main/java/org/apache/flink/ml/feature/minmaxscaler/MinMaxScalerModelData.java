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

package org.apache.flink.ml.feature.minmaxscaler;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputView;
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
 * Model data of {@link MinMaxScalerModel}.
 *
 * <p>This class also provides methods to convert model data from Table to a data stream, and
 * classes to save/load model data.
 */
public class MinMaxScalerModelData {
    public DenseIntDoubleVector minVector;

    public DenseIntDoubleVector maxVector;

    public MinMaxScalerModelData() {}

    public MinMaxScalerModelData(DenseIntDoubleVector minVector, DenseIntDoubleVector maxVector) {
        this.minVector = minVector;
        this.maxVector = maxVector;
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelDataTable The table model data.
     * @return The data stream model data.
     */
    public static DataStream<MinMaxScalerModelData> getModelDataStream(Table modelDataTable) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        return tEnv.toDataStream(modelDataTable)
                .map(
                        x ->
                                new MinMaxScalerModelData(
                                        (DenseIntDoubleVector) x.getField(0),
                                        (DenseIntDoubleVector) x.getField(1)));
    }

    /** Encoder for {@link MinMaxScalerModelData}. */
    public static class ModelDataEncoder implements Encoder<MinMaxScalerModelData> {
        private final DenseIntDoubleVectorSerializer serializer =
                new DenseIntDoubleVectorSerializer();

        @Override
        public void encode(MinMaxScalerModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputView dataOutputView = new DataOutputViewStreamWrapper(outputStream);
            serializer.serialize(modelData.minVector, dataOutputView);
            serializer.serialize(modelData.maxVector, dataOutputView);
        }
    }

    /** Decoder for {@link MinMaxScalerModelData}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<MinMaxScalerModelData> {
        @Override
        public Reader<MinMaxScalerModelData> createReader(
                Configuration config, FSDataInputStream stream) {
            return new Reader<MinMaxScalerModelData>() {
                private final DenseIntDoubleVectorSerializer serializer =
                        new DenseIntDoubleVectorSerializer();

                @Override
                public MinMaxScalerModelData read() throws IOException {
                    DataInputView source = new DataInputViewStreamWrapper(stream);
                    try {
                        DenseIntDoubleVector minVector = serializer.deserialize(source);
                        DenseIntDoubleVector maxVector = serializer.deserialize(source);
                        return new MinMaxScalerModelData(minVector, maxVector);
                    } catch (EOFException e) {
                        return null;
                    }
                }

                @Override
                public void close() throws IOException {
                    stream.close();
                }
            };
        }

        @Override
        public TypeInformation<MinMaxScalerModelData> getProducedType() {
            return TypeInformation.of(MinMaxScalerModelData.class);
        }
    }
}
