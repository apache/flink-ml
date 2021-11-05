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

package org.apache.flink.ml.clustering.kmeans;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/** Provides classes needed for save/load model data. */
public class KMeansModelData {

    public static final Schema SCHEMA =
            Schema.newBuilder()
                    .column("f0", DataTypes.ARRAY(DataTypes.of(DenseVector.class)))
                    .build();

    /** Encoder for the KMeans model data. */
    public static class ModelDataEncoder implements Encoder<DenseVector[]> {
        @Override
        public void encode(DenseVector[] modelData, OutputStream outputStream) {
            Kryo kryo = new Kryo();
            Output output = new Output(outputStream);
            List<double[]> convertedData = new ArrayList<>();
            for (int i = 0; i < modelData.length; i++) {
                convertedData.add(modelData[i].values);
            }
            kryo.writeObject(output, convertedData);
            output.flush();
        }
    }

    /** Decoder for the KMeans model data. */
    public static class ModelDataStreamFormat extends SimpleStreamFormat<DenseVector[]> {
        @Override
        public Reader<DenseVector[]> createReader(Configuration config, FSDataInputStream stream) {
            return new Reader<DenseVector[]>() {
                private final Kryo kryo = new Kryo();
                private final Input input = new Input(stream);

                @Override
                public DenseVector[] read() throws IOException {
                    if (input.eof()) {
                        return null;
                    }
                    ArrayList<double[]> row = kryo.readObject(input, ArrayList.class);

                    DenseVector[] result = new DenseVector[row.size()];
                    for (int i = 0; i < result.length; i++) {
                        result[i] = new DenseVector(row.get(i));
                    }
                    return result;
                }

                @Override
                public void close() throws IOException {
                    stream.close();
                }
            };
        }

        @Override
        public TypeInformation<DenseVector[]> getProducedType() {
            return ObjectArrayTypeInfo.getInfoFor(TypeInformation.of(DenseVector.class));
        }
    }
}
