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

package org.apache.flink.ml.feature.featurehasher;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.table.types.DataType;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.apache.flink.shaded.guava30.com.google.common.hash.Hashing.murmur3_32;

/**
 * A Transformer that transforms a set of categorical or numerical features into a sparse vector of
 * a specified dimension. The rules of hashing categorical columns and numerical columns are as
 * follows:
 *
 * <ul>
 *   <li>For numerical columns, the index of this feature in the output vector is the hash value of
 *       the column name and its correponding value is the same as the input.
 *   <li>For categorical columns, the index of this feature in the output vector is the hash value
 *       of the string "column_name=value" and the corresponding value is 1.0.
 * </ul>
 *
 * <p>If multiple features are projected into the same column, the output values are accumulated.
 * For the hashing trick, see https://en.wikipedia.org/wiki/Feature_hashing for details.
 */
public class FeatureHasher
        implements Transformer<FeatureHasher>, FeatureHasherParams<FeatureHasher> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private static final org.apache.flink.shaded.guava30.com.google.common.hash.HashFunction HASH =
            murmur3_32(0);

    public FeatureHasher() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        ResolvedSchema tableSchema = inputs[0].getResolvedSchema();
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(tableSchema);
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), VectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));
        DataStream<Row> output =
                tEnv.toDataStream(inputs[0])
                        .map(
                                new HashFunction(
                                        getInputCols(),
                                        generateCategoricalCols(
                                                tableSchema, getInputCols(), getCategoricalCols()),
                                        getNumFeatures()),
                                outputTypeInfo);
        Table outputTable = tEnv.fromDataStream(output);
        return new Table[] {outputTable};
    }

    /**
     * The main logic for transforming the categorical and numerical features into a sparse vector.
     * It uses MurMurHash3 to compute the transformed index in the output vector. If multiple
     * features are projected to the same column, their values are accumulated.
     */
    private static class HashFunction implements MapFunction<Row, Row> {
        private final String[] categoricalCols;
        private final int numFeatures;
        private final String[] numericCols;

        public HashFunction(String[] inputCols, String[] categoricalCols, int numFeatures) {
            this.categoricalCols = categoricalCols;
            this.numFeatures = numFeatures;
            this.numericCols = ArrayUtils.removeElements(inputCols, this.categoricalCols);
        }

        @Override
        public Row map(Row row) {
            TreeMap<Integer, Double> feature = new TreeMap<>();
            for (String col : numericCols) {
                if (null != row.getField(col)) {
                    double value = ((Number) row.getFieldAs(col)).doubleValue();
                    updateMap(col, value, feature, numFeatures);
                }
            }
            for (String col : categoricalCols) {
                if (null != row.getField(col)) {
                    updateMap(col + "=" + row.getField(col), 1.0, feature, numFeatures);
                }
            }
            int nnz = feature.size();
            int[] indices = new int[nnz];
            double[] values = new double[nnz];
            int pos = 0;
            for (Map.Entry<Integer, Double> entry : feature.entrySet()) {
                indices[pos] = entry.getKey();
                values[pos] = entry.getValue();
                pos++;
            }
            return Row.join(row, Row.of(new SparseIntDoubleVector(numFeatures, indices, values)));
        }
    }

    private String[] generateCategoricalCols(
            ResolvedSchema tableSchema, String[] inputCols, String[] categoricalCols) {
        if (null == inputCols) {
            return categoricalCols;
        }
        List<String> categoricalList = Arrays.asList(categoricalCols);
        List<String> inputList = Arrays.asList(inputCols);
        if (categoricalCols.length > 0 && !inputList.containsAll(categoricalList)) {
            throw new IllegalArgumentException("CategoricalCols must be included in inputCols!");
        }
        List<DataType> dataColTypes = tableSchema.getColumnDataTypes();
        List<String> dataColNames = tableSchema.getColumnNames();
        List<DataType> inputColTypes = new ArrayList<>();
        for (String col : inputCols) {
            for (int i = 0; i < dataColNames.size(); ++i) {
                if (col.equals(dataColNames.get(i))) {
                    inputColTypes.add(dataColTypes.get(i));
                    break;
                }
            }
        }
        List<String> resultColList = new ArrayList<>();
        for (int i = 0; i < inputCols.length; i++) {
            boolean included = categoricalList.contains(inputCols[i]);
            if (included
                    || DataTypes.BOOLEAN().equals(inputColTypes.get(i))
                    || DataTypes.STRING().equals(inputColTypes.get(i))) {
                resultColList.add(inputCols[i]);
            }
        }
        return resultColList.toArray(new String[0]);
    }

    /**
     * Updates the treeMap which saves the key-value pair of the final vector, use the hash value of
     * the string as key and the accumulate the corresponding value.
     *
     * @param s the string to hash
     * @param value the accumulated value
     */
    private static void updateMap(
            String s, double value, TreeMap<Integer, Double> feature, int numFeature) {
        int hashValue = Math.abs(HASH.hashUnencodedChars(s).asInt());

        int index = Math.floorMod(hashValue, numFeature);
        if (feature.containsKey(index)) {
            feature.put(index, feature.get(index) + value);
        } else {
            feature.put(index, value);
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static FeatureHasher load(StreamTableEnvironment env, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
