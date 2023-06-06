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

package org.apache.flink.ml.feature.elementwiseproduct;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
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

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * A Transformer that multiplies each input vector with a given scaling vector using Hadamard
 * product.
 *
 * <p>If the size of the input vector does not equal the size of the scaling vector, the transformer
 * will throw {@link IllegalArgumentException}.
 */
public class ElementwiseProduct
        implements Transformer<ElementwiseProduct>, ElementwiseProductParams<ElementwiseProduct> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public ElementwiseProduct() {
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
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), VectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));
        DataStream<Row> output =
                tEnv.toDataStream(inputs[0])
                        .map(
                                new ElementwiseProductFunction(getInputCol(), getScalingVec()),
                                outputTypeInfo);
        Table outputTable = tEnv.fromDataStream(output);
        return new Table[] {outputTable};
    }

    private static class ElementwiseProductFunction implements MapFunction<Row, Row> {
        private final String inputCol;
        private final IntDoubleVector scalingVec;

        public ElementwiseProductFunction(String inputCol, IntDoubleVector scalingVec) {
            this.inputCol = inputCol;
            this.scalingVec = scalingVec;
        }

        @Override
        public Row map(Row value) {
            IntDoubleVector inputVec = value.getFieldAs(inputCol);
            if (inputVec != null) {
                if (scalingVec.size() != inputVec.size()) {
                    throw new IllegalArgumentException(
                            "The scaling vector size is "
                                    + scalingVec.size()
                                    + ", which is not equal input vector size("
                                    + inputVec.size()
                                    + ").");
                }
                IntDoubleVector retVec = inputVec.clone();
                BLAS.hDot(scalingVec, retVec);
                return Row.join(value, Row.of(retVec));
            } else {
                return Row.join(value, Row.of((Object) null));
            }
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static ElementwiseProduct load(StreamTableEnvironment env, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
