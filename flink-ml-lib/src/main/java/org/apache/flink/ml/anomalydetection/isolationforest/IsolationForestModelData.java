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

package org.apache.flink.ml.anomalydetection.isolationforest;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.anomalydetection.isolationforest.IsolationForest.IForest;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Model data of {@link IsolationForestModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
public class IsolationForestModelData {

    public IForest iForest;

    public IsolationForestModelData() {}

    public IsolationForestModelData(IForest iForest) {
        this.iForest = iForest;
    }

    public static DataStream<IsolationForestModelData> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(x -> new IsolationForestModelData((IForest) x.getField(0)));
    }

    /** Data encoder for {@link IsolationForestModelData}. */
    public static class ModelDataEncoder implements Encoder<IsolationForestModelData> {
        private final TypeSerializer<IForest> pojoSerializer =
                TypeInformation.of(IForest.class).createSerializer(new ExecutionConfig());

        @Override
        public void encode(IsolationForestModelData modelData, OutputStream outputStream)
                throws IOException {
            pojoSerializer.serialize(
                    modelData.iForest, new DataOutputViewStreamWrapper(outputStream));
        }
    }

    /** Data decoder for {@link IsolationForestModelData}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<IsolationForestModelData> {

        @Override
        public Reader<IsolationForestModelData> createReader(
                Configuration configuration, FSDataInputStream fsDataInputStream)
                throws IOException {
            return new Reader<IsolationForestModelData>() {
                private final TypeSerializer<IForest> pojoSerializer =
                        TypeInformation.of(IForest.class).createSerializer(new ExecutionConfig());

                @Override
                public IsolationForestModelData read() throws IOException {
                    try {
                        DataInputViewStreamWrapper inputViewStreamWrapper =
                                new DataInputViewStreamWrapper(fsDataInputStream);
                        IForest iForest1 = pojoSerializer.deserialize(inputViewStreamWrapper);
                        return new IsolationForestModelData(iForest1);
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
        public TypeInformation<IsolationForestModelData> getProducedType() {
            return TypeInformation.of(IsolationForestModelData.class);
        }
    }
}
