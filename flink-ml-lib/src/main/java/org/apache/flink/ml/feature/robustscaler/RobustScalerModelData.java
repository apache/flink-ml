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

package org.apache.flink.ml.feature.robustscaler;

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
 * Model data of {@link RobustScalerModel}.
 *
 * <p>This class also provides methods to convert model data from Table to a data stream, and
 * classes to save/load model data.
 */
public class RobustScalerModelData {
    public DenseIntDoubleVector medians;

    public DenseIntDoubleVector ranges;

    public RobustScalerModelData() {}

    public RobustScalerModelData(DenseIntDoubleVector medians, DenseIntDoubleVector ranges) {
        this.medians = medians;
        this.ranges = ranges;
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelDataTable The table model data.
     * @return The data stream model data.
     */
    public static DataStream<RobustScalerModelData> getModelDataStream(Table modelDataTable) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        return tEnv.toDataStream(modelDataTable)
                .map(
                        x ->
                                new RobustScalerModelData(
                                        (DenseIntDoubleVector) x.getField("medians"),
                                        (DenseIntDoubleVector) x.getField("ranges")));
    }

    /** Data encoder for the {@link RobustScalerModel} model data. */
    public static class ModelDataEncoder implements Encoder<RobustScalerModelData> {
        private final DenseIntDoubleVectorSerializer serializer =
                new DenseIntDoubleVectorSerializer();

        @Override
        public void encode(RobustScalerModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputViewStreamWrapper outputViewStreamWrapper =
                    new DataOutputViewStreamWrapper(outputStream);
            serializer.serialize(modelData.medians, outputViewStreamWrapper);
            serializer.serialize(modelData.ranges, outputViewStreamWrapper);
        }
    }

    /** Data decoder for the {@link RobustScalerModel} model data. */
    public static class ModelDataDecoder extends SimpleStreamFormat<RobustScalerModelData> {

        @Override
        public Reader<RobustScalerModelData> createReader(
                Configuration configuration, FSDataInputStream inputStream) throws IOException {
            return new Reader<RobustScalerModelData>() {
                private final DenseIntDoubleVectorSerializer serializer =
                        new DenseIntDoubleVectorSerializer();

                @Override
                public RobustScalerModelData read() throws IOException {
                    DataInputViewStreamWrapper inputViewStreamWrapper =
                            new DataInputViewStreamWrapper(inputStream);
                    try {
                        DenseIntDoubleVector medians =
                                serializer.deserialize(inputViewStreamWrapper);
                        DenseIntDoubleVector ranges =
                                serializer.deserialize(inputViewStreamWrapper);
                        return new RobustScalerModelData(medians, ranges);
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
        public TypeInformation<RobustScalerModelData> getProducedType() {
            return TypeInformation.of(RobustScalerModelData.class);
        }
    }
}
