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

package org.apache.flink.ml.feature.onehotencoder;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.common.typeutils.base.MapSerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.api.ModelInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;

/**
 * Model data of {@link OneHotEncoderModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
public class OneHotEncoderModelData implements ModelInfo {
    public Map<Integer, Integer> mapping;
    public long versionId;
    public boolean isLastRecord;

    public OneHotEncoderModelData(Map<Integer, Integer> mapping) {
        this(mapping, System.nanoTime(), true);
    }

    public OneHotEncoderModelData(
            Map<Integer, Integer> mapping, long versionId, boolean isLastRecord) {
        this.mapping = mapping;
        this.versionId = versionId;
        this.isLastRecord = isLastRecord;
    }

    public OneHotEncoderModelData() {}

    @Override
    public long getVersionId() {
        return versionId;
    }

    @Override
    public boolean getIsLastRecord() {
        return isLastRecord;
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<OneHotEncoderModelData> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(
                        (MapFunction<Row, OneHotEncoderModelData>)
                                row ->
                                        new OneHotEncoderModelData(
                                                (Map<Integer, Integer>) row.getField(0),
                                                (long) row.getField(1),
                                                (boolean) row.getField(2)));
    }

    /** Data encoder for the OneHotEncoder model data. */
    public static class ModelDataEncoder implements Encoder<OneHotEncoderModelData> {
        @Override
        public void encode(OneHotEncoderModelData modelData, OutputStream outputStream) {
            DataOutputViewStreamWrapper outputViewStreamWrapper =
                    new DataOutputViewStreamWrapper(outputStream);

            MapSerializer<Integer, Integer> mapSerializer =
                    new MapSerializer<>(IntSerializer.INSTANCE, IntSerializer.INSTANCE);
            try {
                mapSerializer.serialize(modelData.mapping, outputViewStreamWrapper);
                outputViewStreamWrapper.writeLong(modelData.versionId);
                outputViewStreamWrapper.writeBoolean(modelData.isLastRecord);
            } catch (IOException e) {
                e.printStackTrace();
                throw new RuntimeException("OneHot model data sink err.");
            }
        }
    }

    /** Data decoder for the OneHotEncoder model data. */
    public static class ModelDataStreamFormat extends SimpleStreamFormat<OneHotEncoderModelData> {
        @Override
        public Reader<OneHotEncoderModelData> createReader(
                Configuration config, FSDataInputStream stream) {
            return new Reader<OneHotEncoderModelData>() {

                @Override
                public OneHotEncoderModelData read() {
                    try {
                        DataInputViewStreamWrapper inputViewStreamWrapper =
                                new DataInputViewStreamWrapper(stream);

                        MapSerializer<Integer, Integer> mapSerializer =
                                new MapSerializer<>(IntSerializer.INSTANCE, IntSerializer.INSTANCE);
                        Map<Integer, Integer> mapping =
                                mapSerializer.deserialize(inputViewStreamWrapper);

                        return new OneHotEncoderModelData(
                                mapping,
                                inputViewStreamWrapper.readLong(),
                                inputViewStreamWrapper.readBoolean());
                    } catch (IOException e) {
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
        public TypeInformation<OneHotEncoderModelData> getProducedType() {
            return TypeInformation.of(OneHotEncoderModelData.class);
        }
    }
}
