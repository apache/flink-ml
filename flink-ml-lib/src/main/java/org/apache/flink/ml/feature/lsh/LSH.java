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

import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Base class for estimators that support LSH (Locality-sensitive hashing) algorithm for different
 * metrics (e.g., Jaccard distance).
 *
 * <p>The basic idea of LSH is to use a set of hash functions to map input vectors into different
 * buckets, where closer vectors are expected to be in the same bucket with higher probabilities. In
 * detail, each input vector is hashed by all functions. To decide whether two input vectors are
 * mapped into the same bucket, two mechanisms for assigning buckets are proposed as follows.
 *
 * <ul>
 *   <li>AND-amplification: The two input vectors are defined to be in the same bucket as long as
 *       ALL of the hash value matches.
 *   <li>OR-amplification: The two input vectors are defined to be in the same bucket as long as ANY
 *       of the hash value matches.
 * </ul>
 *
 * <p>See: <a
 * href="https://en.wikipedia.org/wiki/Locality-sensitive_hashing">Locality-sensitive_hashing</a>.
 *
 * @param <E> class type of the Estimator implementation.
 * @param <M> class type of the Model this Estimator produces.
 */
abstract class LSH<E extends Estimator<E, M>, M extends LSHModel<M>>
        implements Estimator<E, M>, LSHParams<E> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public LSH() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public M fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Integer> inputDim = getVectorSize(tEnv.toDataStream(inputs[0]), getInputCol());
        return createModel(inputDim, tEnv);
    }

    protected abstract M createModel(DataStream<Integer> inputDim, StreamTableEnvironment tEnv);

    private static DataStream<Integer> getVectorSize(DataStream<Row> input, String vectorCol) {
        DataStream<Integer> vectorSizes =
                input.map(
                        d -> {
                            IntDoubleVector vec = d.getFieldAs(vectorCol);
                            return vec.size();
                        });
        return DataStreamUtils.reduce(
                vectorSizes,
                (s0, s1) -> {
                    Preconditions.checkState(
                            s0.equals(s1), "Vector sizes are not the same: %d %d.", s0, s1);
                    return s0;
                });
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
