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

package org.apache.flink.ml.classification.knn;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Matrix;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.DenseMatrixSerializer;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Model data of {@link KnnModel}.
 *
 * <p>This class also provides methods to convert model data from Table to a data stream, and
 * classes to save/load model data.
 */
public class KnnModelData {

    public DenseMatrix packedFeatures;
    public DenseVector featureNormSquares;
    public DenseVector labels;

    public KnnModelData() {}

    public KnnModelData(
            DenseMatrix packedFeatures, DenseVector featureNormSquares, DenseVector labels) {
        this.packedFeatures = packedFeatures;
        this.featureNormSquares = featureNormSquares;
        this.labels = labels;
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelDataTable The table model data.
     * @return The data stream model data.
     */
    public static DataStream<KnnModelData> getModelDataStream(Table modelDataTable) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        return tEnv.toDataStream(modelDataTable)
                .map(
                        x ->
                                new KnnModelData(
                                        ((Matrix) x.getField(0)).toDense(),
                                        ((Vector) x.getField(1)).toDense(),
                                        ((Vector) x.getField(2)).toDense()));
    }

    /** Encoder for {@link KnnModelData}. */
    public static class ModelDataEncoder implements Encoder<KnnModelData> {
        private final DenseVectorSerializer serializer = new DenseVectorSerializer();

        @Override
        public void encode(KnnModelData modelData, OutputStream outputStream) throws IOException {
            DataOutputView dataOutputView = new DataOutputViewStreamWrapper(outputStream);
            DenseMatrixSerializer.INSTANCE.serialize(modelData.packedFeatures, dataOutputView);
            serializer.serialize(modelData.featureNormSquares, dataOutputView);
            serializer.serialize(modelData.labels, dataOutputView);
        }
    }

    /** Decoder for {@link KnnModelData}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<KnnModelData> {
        @Override
        public Reader<KnnModelData> createReader(Configuration config, FSDataInputStream stream) {
            return new Reader<KnnModelData>() {

                private final DataInputView source = new DataInputViewStreamWrapper(stream);

                private final DenseVectorSerializer serializer = new DenseVectorSerializer();

                @Override
                public KnnModelData read() throws IOException {
                    try {
                        DenseMatrix matrix = DenseMatrixSerializer.INSTANCE.deserialize(source);
                        DenseVector normSquares = serializer.deserialize(source);
                        DenseVector labels = serializer.deserialize(source);
                        return new KnnModelData(matrix, normSquares, labels);
                    } catch (EOFException e) {
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
        public TypeInformation<KnnModelData> getProducedType() {
            return TypeInformation.of(KnnModelData.class);
        }
    }
}
