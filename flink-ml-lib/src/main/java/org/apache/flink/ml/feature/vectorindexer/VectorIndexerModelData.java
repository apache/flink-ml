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

package org.apache.flink.ml.feature.vectorindexer;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.common.typeutils.base.MapSerializer;
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
import java.util.Collections;
import java.util.Map;

/**
 * Model data of {@link VectorIndexerModel}.
 *
 * <p>This class also provides methods to convert model data from Table to DataStream, and classes
 * to save/load model data.
 */
public class VectorIndexerModelData {
    public static final TypeInformation<VectorIndexerModelData> TYPE_INFO =
            Types.POJO(
                    VectorIndexerModelData.class,
                    Collections.singletonMap(
                            "categoryMaps",
                            Types.MAP(Types.INT, Types.MAP(Types.DOUBLE, Types.INT))));

    /**
     * Index of feature values. Keys are column indices. Values are mapping from original continuous
     * features values to 0-based categorical indices. If a feature is not in this map, it is
     * treated as a continuous feature.
     */
    public Map<Integer, Map<Double, Integer>> categoryMaps;

    public VectorIndexerModelData(Map<Integer, Map<Double, Integer>> categoryMaps) {
        this.categoryMaps = categoryMaps;
    }

    public VectorIndexerModelData() {}

    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<VectorIndexerModelData> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(
                        x ->
                                new VectorIndexerModelData(
                                        (Map<Integer, Map<Double, Integer>>) x.getField(0)),
                        TYPE_INFO);
    }

    /** Data encoder for {@link VectorIndexerModel}. */
    public static class ModelDataEncoder implements Encoder<VectorIndexerModelData> {

        @Override
        public void encode(VectorIndexerModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputViewStreamWrapper outputViewStreamWrapper =
                    new DataOutputViewStreamWrapper(outputStream);

            MapSerializer<Integer, Map<Double, Integer>> mapSerializer =
                    new MapSerializer<>(
                            IntSerializer.INSTANCE,
                            new MapSerializer<>(DoubleSerializer.INSTANCE, IntSerializer.INSTANCE));

            mapSerializer.serialize(modelData.categoryMaps, outputViewStreamWrapper);
        }
    }

    /** Data decoder for {@link VectorIndexerModel}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<VectorIndexerModelData> {

        @Override
        public Reader<VectorIndexerModelData> createReader(
                Configuration configuration, FSDataInputStream inputStream) {
            return new Reader<VectorIndexerModelData>() {

                @Override
                public VectorIndexerModelData read() throws IOException {
                    try {
                        DataInputViewStreamWrapper inputViewStreamWrapper =
                                new DataInputViewStreamWrapper(inputStream);
                        MapSerializer<Integer, Map<Double, Integer>> mapSerializer =
                                new MapSerializer<>(
                                        IntSerializer.INSTANCE,
                                        new MapSerializer<>(
                                                DoubleSerializer.INSTANCE, IntSerializer.INSTANCE));
                        Map<Integer, Map<Double, Integer>> categoryMaps =
                                mapSerializer.deserialize(inputViewStreamWrapper);
                        return new VectorIndexerModelData(categoryMaps);
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
        public TypeInformation<VectorIndexerModelData> getProducedType() {
            return TYPE_INFO;
        }
    }
}
