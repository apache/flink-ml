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

package org.apache.flink.ml.feature.imputer;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.api.common.typeutils.base.MapSerializer;
import org.apache.flink.api.common.typeutils.base.StringSerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collections;
import java.util.Map;

/**
 * Model data of {@link ImputerModel}.
 *
 * <p>This class also provides methods to convert model data from Table to a data stream, and
 * classes to save/load model data.
 */
public class ImputerModelData {

    public static final TypeInformation<ImputerModelData> TYPE_INFO =
            Types.POJO(
                    ImputerModelData.class,
                    Collections.singletonMap("surrogates", Types.MAP(Types.STRING, Types.DOUBLE)));

    public Map<String, Double> surrogates;

    public ImputerModelData() {}

    public ImputerModelData(Map<String, Double> surrogates) {
        this.surrogates = surrogates;
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelDataTable The table model data.
     * @return The data stream model data.
     */
    public static DataStream<ImputerModelData> getModelDataStream(Table modelDataTable) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        return tEnv.toDataStream(modelDataTable)
                .map(x -> new ImputerModelData((Map<String, Double>) x.getField(0)), TYPE_INFO);
    }

    /** Encoder for {@link ImputerModelData}. */
    public static class ModelDataEncoder implements Encoder<ImputerModelData> {
        @Override
        public void encode(ImputerModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputView dataOutputView = new DataOutputViewStreamWrapper(outputStream);
            MapSerializer<String, Double> mapSerializer =
                    new MapSerializer<>(StringSerializer.INSTANCE, DoubleSerializer.INSTANCE);
            mapSerializer.serialize(modelData.surrogates, dataOutputView);
        }
    }

    /** Decoder for {@link ImputerModelData}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<ImputerModelData> {

        @Override
        public Reader<ImputerModelData> createReader(
                Configuration configuration, FSDataInputStream fsDataInputStream) {
            return new Reader<ImputerModelData>() {
                @Override
                public ImputerModelData read() throws IOException {
                    DataInputView source = new DataInputViewStreamWrapper(fsDataInputStream);
                    try {
                        MapSerializer<String, Double> mapSerializer =
                                new MapSerializer<>(
                                        StringSerializer.INSTANCE, DoubleSerializer.INSTANCE);
                        Map<String, Double> surrogates = mapSerializer.deserialize(source);
                        return new ImputerModelData(surrogates);
                    } catch (EOFException e) {
                        return null;
                    }
                }

                @Override
                public void close() throws IOException {
                    fsDataInputStream.close();
                }
            };
        }

        @Override
        public TypeInformation<ImputerModelData> getProducedType() {
            return TYPE_INFO;
        }
    }
}
