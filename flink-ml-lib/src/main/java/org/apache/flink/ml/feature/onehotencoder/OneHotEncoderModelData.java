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
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;

import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import java.io.IOException;
import java.io.OutputStream;

/**
 * Model data of {@link OneHotEncoderModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
public class OneHotEncoderModelData {
    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<Tuple2<Integer, Integer>> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(
                        new MapFunction<Row, Tuple2<Integer, Integer>>() {
                            @Override
                            public Tuple2<Integer, Integer> map(Row row) {
                                return new Tuple2<>(
                                        (int) row.getField("f0"), (int) row.getField("f1"));
                            }
                        });
    }

    /** Data encoder for the OneHotEncoder model data. */
    public static class ModelDataEncoder implements Encoder<Tuple2<Integer, Integer>> {
        @Override
        public void encode(Tuple2<Integer, Integer> modelData, OutputStream outputStream) {
            Output output = new Output(outputStream);
            output.writeInt(modelData.f0);
            output.writeInt(modelData.f1);
            output.flush();
        }
    }

    /** Data decoder for the OneHotEncoder model data. */
    public static class ModelDataStreamFormat extends SimpleStreamFormat<Tuple2<Integer, Integer>> {
        @Override
        public Reader<Tuple2<Integer, Integer>> createReader(
                Configuration config, FSDataInputStream stream) {
            return new Reader<Tuple2<Integer, Integer>>() {
                private final Input input = new Input(stream);

                @Override
                public Tuple2<Integer, Integer> read() {
                    if (input.eof()) {
                        return null;
                    }
                    int f0 = input.readInt();
                    int f1 = input.readInt();
                    return new Tuple2<>(f0, f1);
                }

                @Override
                public void close() throws IOException {
                    stream.close();
                }
            };
        }

        @Override
        public TypeInformation<Tuple2<Integer, Integer>> getProducedType() {
            return Types.TUPLE(Types.INT, Types.INT);
        }
    }
}
