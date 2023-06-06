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

package org.apache.flink.ml.feature.dct;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;
import org.jtransforms.dct.DoubleDCT_1D;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * A Transformer that takes the 1D discrete cosine transform of a real vector. No zero padding is
 * performed on the input vector. It returns a real vector of the same length representing the DCT.
 * The return vector is scaled such that the transform matrix is unitary (aka scaled DCT-II).
 *
 * <p>See https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II(DCT-II in Discrete cosine
 * transform).
 */
public class DCT implements Transformer<DCT>, DCTParams<DCT> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public DCT() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(),
                                DenseIntDoubleVectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));

        DataStream<Row> stream =
                tEnv.toDataStream(inputs[0])
                        .map(new DCTFunction(getInputCol(), getInverse()), outputTypeInfo);

        return new Table[] {tEnv.fromDataStream(stream)};
    }

    /**
     * A {@link MapFunction} that contains the main logic to perform discrete cosine transformation.
     */
    private static class DCTFunction implements MapFunction<Row, Row> {
        private final String inputCol;

        private final boolean isInverse;

        private BiConsumer<double[], Boolean> dctTransformer;

        private long previousVectorSize;

        private DCTFunction(String inputCol, boolean isInverse) {
            this.inputCol = inputCol;
            this.isInverse = isInverse;
            this.dctTransformer = null;
            this.previousVectorSize = -1;
        }

        @Override
        public Row map(Row row) throws Exception {
            IntDoubleVector vector = row.getFieldAs(inputCol);

            if (previousVectorSize != vector.size()) {
                if (isInverse) {
                    dctTransformer = new DoubleDCT_1D(vector.size())::inverse;
                } else {
                    dctTransformer = new DoubleDCT_1D(vector.size())::forward;
                }
                previousVectorSize = vector.size();
            }

            double[] array = vector.toArray();
            if (vector instanceof DenseIntDoubleVector) {
                array = array.clone();
            }

            dctTransformer.accept(array, true);

            return Row.join(row, Row.of(Vectors.dense(array)));
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static DCT load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
