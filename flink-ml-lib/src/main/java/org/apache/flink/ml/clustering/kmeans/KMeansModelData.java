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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.util.Preconditions;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Random;

/**
 * Model data of {@link KMeansModel} and {@link OnlineKMeansModel}.
 *
 * <p>This class also provides methods to convert model data from Table to Datastream, and classes
 * to save/load model data.
 */
public class KMeansModelData {

    public DenseVector[] centroids;

    /**
     * The weight of the centroids. It is used when updating the model data in online training
     * process.
     *
     * <p>KMeansModelData objects generated during {@link KMeans#fit(Table...)} also contains this
     * field, so that it can be used as the initial model data of the online training process.
     */
    public DenseVector weights;

    public KMeansModelData(DenseVector[] centroids, DenseVector weights) {
        Preconditions.checkArgument(centroids.length == weights.size());
        this.centroids = centroids;
        this.weights = weights;
    }

    public KMeansModelData() {}

    /**
     * Generates a Table containing a {@link KMeansModelData} instance with randomly generated
     * centroids.
     *
     * @param tEnv The environment where to create the table.
     * @param k The number of generated centroids.
     * @param dim The size of generated centroids.
     * @param weight The weight of the centroids.
     * @param seed Random seed.
     */
    public static Table generateRandomModelData(
            StreamTableEnvironment tEnv, int k, int dim, double weight, long seed) {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);
        return tEnv.fromDataStream(
                env.fromElements(1).map(new RandomCentroidsCreator(k, dim, weight, seed)));
    }

    private static class RandomCentroidsCreator implements MapFunction<Integer, KMeansModelData> {
        private final int k;
        private final int dim;
        private final double weight;
        private final long seed;

        private RandomCentroidsCreator(int k, int dim, double weight, long seed) {
            this.k = k;
            this.dim = dim;
            this.weight = weight;
            this.seed = seed;
        }

        @Override
        public KMeansModelData map(Integer integer) {
            DenseVector[] centroids = new DenseVector[k];
            Random random = new Random(seed);
            for (int i = 0; i < k; i++) {
                centroids[i] = new DenseVector(dim);
                for (int j = 0; j < dim; j++) {
                    centroids[i].values[j] = random.nextDouble();
                }
            }
            DenseVector weights = new DenseVector(k);
            Arrays.fill(weights.values, weight);
            return new KMeansModelData(centroids, weights);
        }
    }

    /**
     * Converts the table model to a data stream.
     *
     * @param modelData The table model data.
     * @return The data stream model data.
     */
    public static DataStream<KMeansModelData> getModelDataStream(Table modelData) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelData).getTableEnvironment();
        return tEnv.toDataStream(modelData)
                .map(
                        x ->
                                new KMeansModelData(
                                        Arrays.stream(((Vector[]) x.getField(0)))
                                                .map(Vector::toDense)
                                                .toArray(DenseVector[]::new),
                                        ((Vector) x.getField(1)).toDense()));
    }

    /** Data encoder for {@link KMeansModelData}. */
    public static class ModelDataEncoder implements Encoder<KMeansModelData> {
        private final DenseVectorSerializer serializer = new DenseVectorSerializer();

        @Override
        public void encode(KMeansModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputViewStreamWrapper outputViewStreamWrapper =
                    new DataOutputViewStreamWrapper(outputStream);
            IntSerializer.INSTANCE.serialize(modelData.centroids.length, outputViewStreamWrapper);
            for (DenseVector denseVector : modelData.centroids) {
                serializer.serialize(denseVector, new DataOutputViewStreamWrapper(outputStream));
            }
            serializer.serialize(modelData.weights, new DataOutputViewStreamWrapper(outputStream));
        }
    }

    /** Data decoder for {@link KMeansModelData}. */
    public static class ModelDataDecoder extends SimpleStreamFormat<KMeansModelData> {
        @Override
        public Reader<KMeansModelData> createReader(
                Configuration config, FSDataInputStream inputStream) {
            return new Reader<KMeansModelData>() {
                private final DenseVectorSerializer serializer = new DenseVectorSerializer();

                @Override
                public KMeansModelData read() throws IOException {
                    try {
                        DataInputViewStreamWrapper inputViewStreamWrapper =
                                new DataInputViewStreamWrapper(inputStream);
                        int numDenseVectors =
                                IntSerializer.INSTANCE.deserialize(inputViewStreamWrapper);
                        DenseVector[] centroids = new DenseVector[numDenseVectors];
                        for (int i = 0; i < numDenseVectors; i++) {
                            centroids[i] = serializer.deserialize(inputViewStreamWrapper);
                        }
                        DenseVector weights = serializer.deserialize(inputViewStreamWrapper);
                        return new KMeansModelData(centroids, weights);
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
        public TypeInformation<KMeansModelData> getProducedType() {
            return TypeInformation.of(KMeansModelData.class);
        }
    }
}
