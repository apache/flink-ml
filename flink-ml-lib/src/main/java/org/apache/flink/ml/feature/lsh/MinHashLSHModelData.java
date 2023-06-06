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

package org.apache.flink.ml.feature.lsh;

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.common.typeutils.base.array.IntPrimitiveArraySerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.util.Preconditions;

import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Random;

/**
 * Model data of {@link MinHashLSHModel}.
 *
 * <p>This class also provides classes to save/load model data.
 */
public class MinHashLSHModelData extends LSHModelData {

    // A large prime smaller than sqrt(2^63 − 1).
    private static final int HASH_PRIME = 2038074743;

    public int numHashTables;
    public int numHashFunctionsPerTable;
    public int[] randCoefficientA;
    public int[] randCoefficientB;

    public MinHashLSHModelData() {}

    public MinHashLSHModelData(
            int numHashTables,
            int numHashFunctionsPerTable,
            int[] randCoefficientA,
            int[] randCoefficientB) {
        this.numHashTables = numHashTables;
        this.numHashFunctionsPerTable = numHashFunctionsPerTable;
        this.randCoefficientA = randCoefficientA;
        this.randCoefficientB = randCoefficientB;
    }

    public static MinHashLSHModelData generateModelData(
            int numHashTables, int numHashFunctionsPerTable, int dim, long seed) {
        Preconditions.checkArgument(
                dim <= HASH_PRIME,
                "The input vector dimension %d exceeds the threshold %s.",
                dim,
                HASH_PRIME);

        Random random = new Random(seed);
        int numHashFunctions = numHashTables * numHashFunctionsPerTable;
        int[] randCoeffA = new int[numHashFunctions];
        int[] randCoeffB = new int[numHashFunctions];
        for (int i = 0; i < numHashFunctions; i += 1) {
            randCoeffA[i] = 1 + random.nextInt(HASH_PRIME - 1);
            randCoeffB[i] = random.nextInt(HASH_PRIME - 1);
        }
        return new MinHashLSHModelData(
                numHashTables, numHashFunctionsPerTable, randCoeffA, randCoeffB);
    }

    static class ModelDataDecoder extends SimpleStreamFormat<MinHashLSHModelData> {
        @Override
        public Reader<MinHashLSHModelData> createReader(
                Configuration configuration, FSDataInputStream fsDataInputStream)
                throws IOException {
            return new Reader<MinHashLSHModelData>() {
                @Override
                public MinHashLSHModelData read() throws IOException {
                    try {
                        DataInputViewStreamWrapper source =
                                new DataInputViewStreamWrapper(fsDataInputStream);
                        int numHashTables = IntSerializer.INSTANCE.deserialize(source);
                        int numHashFunctionsPerTable = IntSerializer.INSTANCE.deserialize(source);
                        int[] randCoeffA = IntPrimitiveArraySerializer.INSTANCE.deserialize(source);
                        int[] randCoeffB = IntPrimitiveArraySerializer.INSTANCE.deserialize(source);
                        return new MinHashLSHModelData(
                                numHashTables, numHashFunctionsPerTable, randCoeffA, randCoeffB);
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
        public TypeInformation<MinHashLSHModelData> getProducedType() {
            return TypeInformation.of(MinHashLSHModelData.class);
        }
    }

    @Override
    public DenseIntDoubleVector[] hashFunction(IntDoubleVector vec) {
        int[] indices = vec.toSparse().indices;
        Preconditions.checkArgument(indices.length > 0, "Must have at least 1 non zero entry.");
        double[][] hashValues = new double[numHashTables][numHashFunctionsPerTable];
        for (int i = 0; i < numHashTables; i += 1) {
            for (int j = 0; j < numHashFunctionsPerTable; j += 1) {
                // For each hash function, the hash value is computed by
                // min(((1 + index) * randCoefficientA + randCoefficientB) % HASH_PRIME).
                int coeffA = randCoefficientA[i * numHashFunctionsPerTable + j];
                int coeffB = randCoefficientB[i * numHashFunctionsPerTable + j];
                long minv = HASH_PRIME;
                for (int index : indices) {
                    minv = Math.min(minv, ((1L + index) * coeffA + coeffB) % HASH_PRIME);
                }
                hashValues[i][j] = minv;
            }
        }
        return Arrays.stream(hashValues)
                .map(DenseIntDoubleVector::new)
                .toArray(DenseIntDoubleVector[]::new);
    }

    @Override
    public double keyDistance(IntDoubleVector x, IntDoubleVector y) {
        int[] xIndices = x.toSparse().indices;
        int[] yIndices = y.toSparse().indices;
        Preconditions.checkArgument(
                xIndices.length + yIndices.length > 0,
                "The union of two input sets must have at least 1 elements");
        int px = 0, py = 0;
        int intersectionSize = 0;
        while (px < xIndices.length && py < yIndices.length) {
            if (xIndices[px] == yIndices[py]) {
                intersectionSize += 1;
                px += 1;
                py += 1;
            } else if (xIndices[px] < yIndices[py]) {
                px += 1;
            } else {
                py += 1;
            }
        }
        int unionSize = xIndices.length + yIndices.length - intersectionSize;
        return 1. - 1. * intersectionSize / unionSize;
    }

    /** Encoder for {@link MinHashLSHModelData}. */
    public static class ModelDataEncoder implements Encoder<MinHashLSHModelData> {

        @Override
        public void encode(MinHashLSHModelData modelData, OutputStream outputStream)
                throws IOException {
            DataOutputView dataOutputView = new DataOutputViewStreamWrapper(outputStream);
            IntSerializer.INSTANCE.serialize(modelData.numHashTables, dataOutputView);
            IntSerializer.INSTANCE.serialize(modelData.numHashFunctionsPerTable, dataOutputView);
            IntPrimitiveArraySerializer.INSTANCE.serialize(
                    modelData.randCoefficientA, dataOutputView);
            IntPrimitiveArraySerializer.INSTANCE.serialize(
                    modelData.randCoefficientB, dataOutputView);
        }
    }
}
