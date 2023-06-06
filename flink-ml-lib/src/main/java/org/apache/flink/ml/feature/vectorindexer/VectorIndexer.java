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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * An Estimator which implements the vector indexing algorithm.
 *
 * <p>A vector indexer maps each column of the input vector into a continuous/categorical feature.
 * Whether one feature is transformed into a continuous or categorical feature depends on the number
 * of distinct values in this column. If the number of distinct values in one column is greater than
 * a specified parameter (i.e., maxCategories), the corresponding output column is unchanged.
 * Otherwise, it is transformed into a categorical value. For categorical outputs, the indices are
 * in [0, numDistinctValuesInThisColumn].
 *
 * <p>The output model is organized in ascending order except that 0.0 is always mapped to 0 (for
 * sparsity). We list two examples here:
 *
 * <ul>
 *   <li>If one column contains {-1.0, 1.0}, then -1.0 should be encoded as 0 and 1.0 will be
 *       encoded as 1.
 *   <li>If one column contains {-1.0, 0.0, 1.0}, then -1.0 should be encoded as 1, 0.0 should be
 *       encoded as 0 and 1.0 should be encoded as 2.
 * </ul>
 *
 * <p>The `keep` option of {@link HasHandleInvalid} means that we put the invalid entries in a
 * special bucket, whose index is the number of distinct values in this column.
 */
public class VectorIndexer
        implements Estimator<VectorIndexer, VectorIndexerModel>,
                VectorIndexerParams<VectorIndexer> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public VectorIndexer() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public VectorIndexerModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        int maxCategories = getMaxCategories();
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<List<Double>[]> localDistinctDoubles =
                tEnv.toDataStream(inputs[0])
                        .transform(
                                "computeDistinctDoublesOperator",
                                Types.OBJECT_ARRAY(Types.LIST(Types.DOUBLE)),
                                new ComputeDistinctDoublesOperator(getInputCol(), maxCategories));

        DataStream<List<Double>[]> distinctDoubles =
                DataStreamUtils.reduce(
                        localDistinctDoubles,
                        (ReduceFunction<List<Double>[]>)
                                (value1, value2) -> {
                                    for (int i = 0; i < value1.length; i++) {
                                        if (value1[i] == null || value2[i] == null) {
                                            value1[i] = null;
                                        } else {
                                            HashSet<Double> tmp = new HashSet<>(value1[i]);
                                            tmp.addAll(value2[i]);
                                            value1[i] = new ArrayList<>(tmp);
                                        }
                                    }
                                    return value1;
                                });

        DataStream<VectorIndexerModelData> modelData =
                distinctDoubles.map(
                        new ModelGenerator(maxCategories), VectorIndexerModelData.TYPE_INFO);
        modelData.getTransformation().setParallelism(1);

        Schema schema =
                Schema.newBuilder()
                        .column(
                                "categoryMaps",
                                DataTypes.MAP(
                                        DataTypes.INT(),
                                        DataTypes.MAP(DataTypes.DOUBLE(), DataTypes.INT())))
                        .build();

        VectorIndexerModel model =
                new VectorIndexerModel().setModelData(tEnv.fromDataStream(modelData, schema));
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static VectorIndexer load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Computes the distinct doubles by columns. If the number of distinct values in one column is
     * greater than maxCategories, the corresponding returned HashSet is null.
     */
    private static class ComputeDistinctDoublesOperator
            extends AbstractStreamOperator<List<Double>[]>
            implements OneInputStreamOperator<Row, List<Double>[]>, BoundedOneInput {
        /** The name of input column. */
        private final String inputCol;
        /** Max number of categories. */
        private final int maxCategories;
        /** The distinct doubles of each column. */
        private HashSet<Double>[] doublesByColumn;
        /** The state of doublesByColumn. */
        private ListState<List<Double>[]> doublesByColumnState;

        public ComputeDistinctDoublesOperator(String inputCol, int maxCategories) {
            this.inputCol = inputCol;
            this.maxCategories = maxCategories;
        }

        @Override
        public void endInput() {
            if (doublesByColumn != null) {
                output.collect(new StreamRecord<>(convertToListArray(doublesByColumn)));
            }
            doublesByColumnState.clear();
        }

        @Override
        public void processElement(StreamRecord<Row> element) {
            if (doublesByColumn == null) {
                // First record.
                IntDoubleVector vector = (IntDoubleVector) element.getValue().getField(inputCol);
                doublesByColumn = new HashSet[vector.size()];
                for (int i = 0; i < doublesByColumn.length; i++) {
                    doublesByColumn[i] = new HashSet<>();
                }
            }

            IntDoubleVector vector = (IntDoubleVector) element.getValue().getField(inputCol);
            Preconditions.checkState(
                    vector.size() == doublesByColumn.length,
                    "The size of the all input vectors should be the same.");
            double[] values = vector.toDense().values;
            for (int i = 0; i < values.length; i++) {
                if (doublesByColumn[i] != null) {
                    doublesByColumn[i].add(values[i]);
                    if (doublesByColumn[i].size() > maxCategories) {
                        doublesByColumn[i] = null;
                    }
                }
            }
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            doublesByColumnState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "doublesByColumnState",
                                            Types.OBJECT_ARRAY(Types.LIST(Types.DOUBLE))));

            OperatorStateUtils.getUniqueElement(doublesByColumnState, "doublesByColumnState")
                    .ifPresent(x -> doublesByColumn = convertToHashSetArray(x));
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            if (doublesByColumn != null) {
                doublesByColumnState.update(
                        Collections.singletonList(convertToListArray(doublesByColumn)));
            }
        }

        private List<Double>[] convertToListArray(HashSet<Double>[] array) {
            List<Double>[] results = new ArrayList[array.length];
            for (int i = 0; i < array.length; i++) {
                results[i] = new ArrayList<>(array[i]);
            }
            return results;
        }

        private HashSet<Double>[] convertToHashSetArray(List<Double>[] array) {
            HashSet<Double>[] results = new HashSet[array.length];
            for (int i = 0; i < array.length; i++) {
                results[i] = new HashSet<>(array[i]);
            }
            return results;
        }
    }

    /**
     * Merges all the distinct doubles by columns and generates the {@link VectorIndexerModelData}.
     */
    private static class ModelGenerator
            implements MapFunction<List<Double>[], VectorIndexerModelData> {
        private final int maxCategories;

        public ModelGenerator(int maxCategories) {
            this.maxCategories = maxCategories;
        }

        @Override
        public VectorIndexerModelData map(List<Double>[] distinctDoubles) {
            Map<Integer, Map<Double, Integer>> categoryMaps = new HashMap<>();
            for (int i = 0; i < distinctDoubles.length; i++) {
                if (distinctDoubles[i] != null && distinctDoubles[i].size() <= maxCategories) {
                    double[] values =
                            distinctDoubles[i].stream().mapToDouble(Double::doubleValue).toArray();
                    Arrays.sort(values);
                    // If 0 exists, we put it as the first element.
                    int index0 = Arrays.binarySearch(values, 0);
                    while (index0 > 0) {
                        values[index0] = values[--index0];
                    }
                    if (index0 == 0) {
                        values[index0] = 0;
                    }
                    Map<Double, Integer> valueAndIndex = new HashMap<>(values.length);
                    for (int valueIdx = 0; valueIdx < values.length; valueIdx++) {
                        valueAndIndex.put(values[valueIdx], valueIdx);
                    }
                    categoryMaps.put(i, valueAndIndex);
                }
            }

            return new VectorIndexerModelData(categoryMaps);
        }
    }
}
