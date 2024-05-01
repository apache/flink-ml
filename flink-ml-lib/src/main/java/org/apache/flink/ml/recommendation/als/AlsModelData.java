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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Model data of {@link AlsModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
public class AlsModelData {

    public List<Tuple2<Long, float[]>> userFactors;
    public List<Tuple2<Long, float[]>> itemFactors;

    public AlsModelData(
            List<Tuple2<Long, float[]>> userFactors, List<Tuple2<Long, float[]>> itemFactors) {
        this.userFactors = userFactors;
        this.itemFactors = itemFactors;
    }

    public AlsModelData(AlsModelData modelData) {
        this.userFactors = modelData.userFactors;
        this.itemFactors = modelData.itemFactors;
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<AlsModelData> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData).map(AlsModelData::parseModel);
    }

    private static AlsModelData parseModel(Row modelRow) {
        return new AlsModelData(modelRow.getFieldAs(0));
    }

    /** Data encoder for {@link AlsModel}. */
    public static class ModelDataEncoder implements Encoder<AlsModelData> {

        @Override
        public void encode(AlsModelData modelData, OutputStream outputStream) throws IOException {

            DataOutputView dataOutputView = new DataOutputViewStreamWrapper(outputStream);
            dataOutputView.writeInt(modelData.userFactors.size());
            if (modelData.userFactors.size() > 0) {
                dataOutputView.writeInt(modelData.userFactors.get(0).f1.length);
                for (int i = 0; i < modelData.userFactors.size(); ++i) {
                    dataOutputView.writeLong(modelData.userFactors.get(i).f0);
                    float[] values = modelData.userFactors.get(i).f1;
                    for (float value : values) {
                        dataOutputView.writeFloat(value);
                    }
                }
            }
            dataOutputView.writeInt(modelData.itemFactors.size());
            if (modelData.itemFactors.size() > 0) {
                dataOutputView.writeInt(modelData.itemFactors.get(0).f1.length);
                for (int i = 0; i < modelData.itemFactors.size(); ++i) {
                    dataOutputView.writeLong(modelData.itemFactors.get(i).f0);
                    float[] values = modelData.itemFactors.get(i).f1;
                    for (float value : values) {
                        dataOutputView.writeFloat(value);
                    }
                }
            }
        }
    }

    /** Data decoder for {@link AlsModel}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<AlsModelData> {

        @Override
        public Reader<AlsModelData> createReader(
                Configuration configuration, FSDataInputStream inputStream) {
            return new Reader<AlsModelData>() {

                @Override
                public AlsModelData read() throws IOException {
                    try {
                        DataInputViewStreamWrapper inputViewStreamWrapper =
                                new DataInputViewStreamWrapper(inputStream);
                        int sizeUser = inputViewStreamWrapper.readInt();
                        List<Tuple2<Long, float[]>> userFactors = new ArrayList<>(sizeUser);

                        if (sizeUser > 0) {
                            int rank = inputViewStreamWrapper.readInt();
                            for (int i = 0; i < sizeUser; ++i) {
                                long id = inputViewStreamWrapper.readLong();
                                float[] factors = new float[rank];
                                for (int j = 0; j < rank; ++j) {
                                    factors[j] = inputViewStreamWrapper.readFloat();
                                }
                                userFactors.add(Tuple2.of(id, factors));
                            }
                        }
                        int sizeItem = inputViewStreamWrapper.readInt();
                        List<Tuple2<Long, float[]>> itemFactors = new ArrayList<>(sizeItem);
                        if (sizeItem > 0) {
                            int rank = inputViewStreamWrapper.readInt();
                            for (int i = 0; i < sizeItem; ++i) {
                                long id = inputViewStreamWrapper.readLong();
                                float[] factors = new float[rank];
                                for (int j = 0; j < rank; ++j) {
                                    factors[j] = inputViewStreamWrapper.readFloat();
                                }
                                itemFactors.add(Tuple2.of(id, factors));
                            }
                        }
                        return new AlsModelData(userFactors, itemFactors);
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
        public TypeInformation<AlsModelData> getProducedType() {
            return TypeInformation.of(AlsModelData.class);
        }
    }
}
