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

package org.apache.flink.ml.feature.countvectorizer;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.base.array.StringArraySerializer;
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

/**
 * Model data of {@link CountVectorizerModel}.
 *
 * <p>This class also provides methods to convert model data from Table to a data stream, and
 * classes to save/load model data.
 */
public class CountVectorizerModelData {

    /** The array over terms, only the terms in the vocabulary will be counted. */
    public String[] vocabulary;

    public CountVectorizerModelData() {}

    public CountVectorizerModelData(String[] vocabulary) {
        this.vocabulary = vocabulary;
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelDataTable The table model data.
     * @return The data stream model data.
     */
    public static DataStream<CountVectorizerModelData> getModelDataStream(Table modelDataTable) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        return tEnv.toDataStream(modelDataTable)
                .map(x -> new CountVectorizerModelData((String[]) x.getField(0)));
    }

    /** Encoder for {@link CountVectorizerModelData}. */
    public static class ModelDataEncoder implements Encoder<CountVectorizerModelData> {
        @Override
        public void encode(CountVectorizerModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputView dataOutputView = new DataOutputViewStreamWrapper(outputStream);
            StringArraySerializer.INSTANCE.serialize(modelData.vocabulary, dataOutputView);
        }
    }

    /** Decoder for {@link CountVectorizerModelData}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<CountVectorizerModelData> {

        @Override
        public Reader<CountVectorizerModelData> createReader(
                Configuration configuration, FSDataInputStream fsDataInputStream) {
            return new Reader<CountVectorizerModelData>() {
                @Override
                public CountVectorizerModelData read() throws IOException {
                    DataInputView source = new DataInputViewStreamWrapper(fsDataInputStream);
                    try {
                        String[] vocabulary = StringArraySerializer.INSTANCE.deserialize(source);
                        return new CountVectorizerModelData(vocabulary);
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
        public TypeInformation<CountVectorizerModelData> getProducedType() {
            return TypeInformation.of(CountVectorizerModelData.class);
        }
    }
}
