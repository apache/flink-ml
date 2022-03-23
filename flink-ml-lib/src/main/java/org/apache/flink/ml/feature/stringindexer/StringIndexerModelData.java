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

package org.apache.flink.ml.feature.stringindexer;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.common.typeutils.base.array.StringArraySerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Model data of {@link StringIndexerModel} and {@link IndexToStringModel}.
 *
 * <p>This class also provides methods to convert model data from Table to DataStream, and classes
 * to save/load model data.
 */
public class StringIndexerModelData {
    /** Ordered strings of each input column. */
    public String[][] stringArrays;

    public StringIndexerModelData(String[][] stringArrays) {
        this.stringArrays = stringArrays;
    }

    public StringIndexerModelData() {}

    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<StringIndexerModelData> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(x -> new StringIndexerModelData((String[][]) x.getField(0)));
    }

    /** Data encoder for {@link StringIndexerModel} and {@link IndexToStringModel}. */
    public static class ModelDataEncoder implements Encoder<StringIndexerModelData> {

        @Override
        public void encode(StringIndexerModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputViewStreamWrapper outputViewStreamWrapper =
                    new DataOutputViewStreamWrapper(outputStream);

            IntSerializer.INSTANCE.serialize(
                    modelData.stringArrays.length, outputViewStreamWrapper);
            for (String[] strings : modelData.stringArrays) {
                StringArraySerializer.INSTANCE.serialize(strings, outputViewStreamWrapper);
            }
        }
    }

    /** Data decoder for {@link StringIndexerModel} and {@link IndexToStringModel}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<StringIndexerModelData> {

        @Override
        public Reader<StringIndexerModelData> createReader(
                Configuration configuration, FSDataInputStream inputStream) {
            return new Reader<StringIndexerModelData>() {

                @Override
                public StringIndexerModelData read() throws IOException {
                    try {
                        DataInputViewStreamWrapper inputViewStreamWrapper =
                                new DataInputViewStreamWrapper(inputStream);

                        int numCols = IntSerializer.INSTANCE.deserialize(inputViewStreamWrapper);
                        String[][] stringsArray = new String[numCols][];
                        for (int i = 0; i < numCols; i++) {
                            stringsArray[i] =
                                    StringArraySerializer.INSTANCE.deserialize(
                                            inputViewStreamWrapper);
                        }

                        return new StringIndexerModelData(stringsArray);
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
        public TypeInformation<StringIndexerModelData> getProducedType() {
            return TypeInformation.of(StringIndexerModelData.class);
        }
    }
}
