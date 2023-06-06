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

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.EndOfStreamWindows;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.typeinfo.PriorityQueueTypeInfo;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Base class for LSH model.
 *
 * <p>In addition to transforming input feature vectors to multiple hash values, it also supports
 * approximate nearest neighbors search within a dataset regarding a key vector and approximate
 * similarity join between two datasets.
 *
 * @param <T> class type of the LSHModel implementation itself.
 */
abstract class LSHModel<T extends LSHModel<T>> implements Model<T>, LSHModelParams<T> {
    private static final String MODEL_DATA_BC_KEY = "modelData";

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    /** Stores the corresponding model data class of T. */
    private final Class<? extends LSHModelData> modelDataClass;

    protected Table modelDataTable;

    public LSHModel(Class<? extends LSHModelData> modelDataClass) {
        this.modelDataClass = modelDataClass;
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public T setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
        return (T) this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<? extends LSHModelData> modelData =
                tEnv.toDataStream(modelDataTable, modelDataClass);

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        TypeInformation<?> outputType = TypeInformation.of(DenseIntDoubleVector[].class);
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), outputType),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(tEnv.toDataStream(inputs[0])),
                        Collections.singletonMap(MODEL_DATA_BC_KEY, modelData),
                        inputList -> {
                            //noinspection unchecked
                            DataStream<Row> data = (DataStream<Row>) inputList.get(0);
                            return data.map(new PredictFunction(getInputCol()), outputTypeInfo);
                        });
        return new Table[] {tEnv.fromDataStream(output)};
    }

    /**
     * Approximately finds at most k items from a dataset which have the closest distance to a given
     * item. If the `outputCol` is missing in the given dataset, this method transforms the dataset
     * with the model at first.
     *
     * @param dataset The dataset in which to to search for nearest neighbors.
     * @param key The item to search for.
     * @param k The maximum number of nearest neighbors.
     * @param distCol The output column storing the distance between each neighbor and the key.
     * @return A dataset containing at most k items closest to the key with a column named `distCol`
     *     appended.
     */
    public Table approxNearestNeighbors(Table dataset, IntDoubleVector key, int k, String distCol) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) dataset).getTableEnvironment();
        Table transformedTable =
                (dataset.getResolvedSchema().getColumnNames().contains(getOutputCol()))
                        ? dataset
                        : transform(dataset)[0];

        DataStream<? extends LSHModelData> modelData =
                tEnv.toDataStream(modelDataTable, modelDataClass);

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(transformedTable.getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), Types.DOUBLE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), distCol));

        // Fetches items in the same bucket with key's, and calculates their distances to key.
        DataStream<Row> filteredData =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(tEnv.toDataStream(transformedTable)),
                        Collections.singletonMap(MODEL_DATA_BC_KEY, modelData),
                        inputList -> {
                            //noinspection unchecked
                            DataStream<Row> data = (DataStream<Row>) inputList.get(0);
                            return data.flatMap(
                                    new FilterByBucketFunction(getInputCol(), getOutputCol(), key),
                                    outputTypeInfo);
                        });
        TopKFunction topKFunction = new TopKFunction(distCol, k);
        DataStream<List<Row>> topKList =
                DataStreamUtils.aggregate(
                        filteredData,
                        topKFunction,
                        new PriorityQueueTypeInfo(topKFunction.getComparator(), outputTypeInfo),
                        Types.LIST(outputTypeInfo));
        DataStream<Row> topKData =
                topKList.flatMap(
                        (value, out) -> {
                            for (Row row : value) {
                                out.collect(row);
                            }
                        });
        topKData.getTransformation().setOutputType(outputTypeInfo);
        return tEnv.fromDataStream(topKData);
    }

    /**
     * An overloaded version of `approxNearestNeighbors` with "distCol" as default value of
     * `distCol`.
     */
    public Table approxNearestNeighbors(Table dataset, IntDoubleVector key, int k) {
        return approxNearestNeighbors(dataset, key, k, "distCol");
    }

    /**
     * Joins two datasets to approximately find all pairs of rows whose distance are smaller than or
     * equal to the threshold. If the `outputCol` is missing in either dataset, this method
     * transforms the dataset at first.
     *
     * @param datasetA One dataset.
     * @param datasetB The other dataset.
     * @param threshold The distance threshold.
     * @param idCol A column in the two datasets to identify each row.
     * @param distCol The output column storing the distance between each pair of rows.
     * @return A joined dataset containing pairs of rows. The original rows are in columns
     *     "datasetA" and "datasetB", and a column "distCol" is added to show the distance between
     *     each pair.
     */
    public Table approxSimilarityJoin(
            Table datasetA, Table datasetB, double threshold, String idCol, String distCol) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) datasetA).getTableEnvironment();

        DataStream<Row> explodedA = preprocessData(datasetA, idCol);
        DataStream<Row> explodedB = preprocessData(datasetB, idCol);

        RowTypeInfo inputTypeInfo = getOutputType(datasetA, idCol);
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        inputTypeInfo.getTypeAt(0),
                        inputTypeInfo.getTypeAt(0),
                        inputTypeInfo.getTypeAt(1),
                        inputTypeInfo.getTypeAt(1));

        DataStream<? extends LSHModelData> modelData =
                tEnv.toDataStream(modelDataTable, modelDataClass);
        DataStream<Row> sameBucketPairs =
                explodedA
                        .join(explodedB)
                        .where(new IndexHashValueKeySelector())
                        .equalTo(new IndexHashValueKeySelector())
                        .window(EndOfStreamWindows.get())
                        .apply(
                                (r0, r1) ->
                                        Row.of(
                                                r0.getField(0),
                                                r1.getField(0),
                                                r0.getField(1),
                                                r1.getField(1)),
                                outputTypeInfo);

        DataStream<Row> distinctSameBucketPairs =
                DataStreamUtils.reduce(
                        sameBucketPairs.keyBy(
                                new KeySelector<Row, Tuple2<Integer, Integer>>() {
                                    @Override
                                    public Tuple2<Integer, Integer> getKey(Row r) {
                                        return Tuple2.of(r.getFieldAs(0), r.getFieldAs(1));
                                    }
                                }),
                        (r0, r1) -> r0,
                        outputTypeInfo);

        TypeInformation<?> idColType =
                TableUtils.getRowTypeInfo(datasetA.getResolvedSchema()).getTypeAt(idCol);
        DataStream<Row> pairsWithDists =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(distinctSameBucketPairs),
                        Collections.singletonMap(MODEL_DATA_BC_KEY, modelData),
                        inputList -> {
                            DataStream<Row> data = (DataStream<Row>) inputList.get(0);
                            return data.flatMap(
                                    new FilterByDistanceFunction(threshold),
                                    new RowTypeInfo(
                                            new TypeInformation[] {
                                                idColType, idColType, Types.DOUBLE
                                            },
                                            new String[] {"datasetA.id", "datasetB.id", distCol}));
                        });
        return tEnv.fromDataStream(pairsWithDists);
    }

    /**
     * An overloaded version of `approxNearestNeighbors` with "distCol" as default value of
     * `distCol`.
     */
    public Table approxSimilarityJoin(
            Table datasetA, Table datasetB, double threshold, String idCol) {
        return approxSimilarityJoin(datasetA, datasetB, threshold, idCol, "distCol");
    }

    private DataStream<Row> preprocessData(Table dataTable, String idCol) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) dataTable).getTableEnvironment();

        dataTable =
                (dataTable.getResolvedSchema().getColumnNames().contains(getOutputCol()))
                        ? dataTable
                        : transform(dataTable)[0];
        RowTypeInfo outputTypeInfo = getOutputType(dataTable, idCol);

        return tEnv.toDataStream(dataTable)
                .flatMap(
                        new ExplodeHashValuesFunction(idCol, getInputCol(), getOutputCol()),
                        outputTypeInfo);
    }

    private RowTypeInfo getOutputType(Table dataTable, String idCol) {
        final String indexCol = "index";
        final String hashValueCol = "hashValue";
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(dataTable.getResolvedSchema());
        TypeInformation<?> idColType = inputTypeInfo.getTypeAt(idCol);

        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        new TypeInformation[] {
                            idColType,
                            VectorTypeInfo.INSTANCE,
                            Types.INT,
                            DenseIntDoubleVectorTypeInfo.INSTANCE
                        },
                        new String[] {idCol, getInputCol(), indexCol, hashValueCol});
        return outputTypeInfo;
    }

    private static class PredictFunction extends RichMapFunction<Row, Row> {
        private final String inputCol;

        private LSHModelData modelData;

        public PredictFunction(String inputCol) {
            this.inputCol = inputCol;
        }

        @Override
        public Row map(Row value) throws Exception {
            if (null == modelData) {
                modelData =
                        (LSHModelData)
                                getRuntimeContext().getBroadcastVariable(MODEL_DATA_BC_KEY).get(0);
            }
            IntDoubleVector[] hashValues = modelData.hashFunction(value.getFieldAs(inputCol));
            return Row.join(value, Row.of((Object) hashValues));
        }
    }

    private static class FilterByBucketFunction extends RichFlatMapFunction<Row, Row> {
        private final String inputCol;
        private final String outputCol;
        private final IntDoubleVector key;
        private LSHModelData modelData;
        private DenseIntDoubleVector[] keyHashes;

        public FilterByBucketFunction(String inputCol, String outputCol, IntDoubleVector key) {
            this.inputCol = inputCol;
            this.outputCol = outputCol;
            this.key = key;
        }

        @Override
        public void flatMap(Row value, Collector<Row> out) throws Exception {
            if (null == modelData) {
                modelData =
                        (LSHModelData)
                                getRuntimeContext().getBroadcastVariable(MODEL_DATA_BC_KEY).get(0);
                keyHashes = modelData.hashFunction(key);
            }
            DenseIntDoubleVector[] hashes = value.getFieldAs(outputCol);
            boolean sameBucket = false;
            for (int i = 0; i < keyHashes.length; i += 1) {
                if (keyHashes[i].equals(hashes[i])) {
                    sameBucket = true;
                    break;
                }
            }
            if (!sameBucket) {
                return;
            }
            IntDoubleVector vec = value.getFieldAs(inputCol);
            double dist = modelData.keyDistance(key, vec);
            out.collect(Row.join(value, Row.of(dist)));
        }
    }

    private static class TopKFunction
            implements AggregateFunction<Row, PriorityQueue<Row>, List<Row>> {
        private final int numNearestNeighbors;
        private final String distCol;

        private static class DistColComparator implements Comparator<Row>, Serializable {

            private final String distCol;

            private DistColComparator(String distCol) {
                this.distCol = distCol;
            }

            @Override
            public int compare(Row o1, Row o2) {
                return Double.compare(o1.getFieldAs(distCol), o2.getFieldAs(distCol));
            }
        }

        public TopKFunction(String distCol, int numNearestNeighbors) {
            this.distCol = distCol;
            this.numNearestNeighbors = numNearestNeighbors;
        }

        @Override
        public PriorityQueue<Row> createAccumulator() {
            return new PriorityQueue<>(numNearestNeighbors, getComparator());
        }

        @Override
        public PriorityQueue<Row> add(Row value, PriorityQueue<Row> accumulator) {
            if (accumulator.size() == numNearestNeighbors) {
                Row peek = accumulator.peek();
                if (accumulator.comparator().compare(value, peek) < 0) {
                    accumulator.poll();
                }
            }
            accumulator.add(value);
            return accumulator;
        }

        @Override
        public List<Row> getResult(PriorityQueue<Row> accumulator) {
            return new ArrayList<>(accumulator);
        }

        @Override
        public PriorityQueue<Row> merge(PriorityQueue<Row> a, PriorityQueue<Row> b) {
            PriorityQueue<Row> merged = new PriorityQueue<>(a);
            for (Row row : b) {
                add(row, merged);
            }
            return merged;
        }

        private Comparator<Row> getComparator() {
            return new DistColComparator(distCol);
        }
    }

    private static class ExplodeHashValuesFunction implements FlatMapFunction<Row, Row> {
        private final String idCol;
        private final String inputCol;
        private final String outputCol;

        public ExplodeHashValuesFunction(String idCol, String inputCol, String outputCol) {
            this.idCol = idCol;
            this.inputCol = inputCol;
            this.outputCol = outputCol;
        }

        @Override
        public void flatMap(Row value, Collector<Row> out) throws Exception {
            Row kept = Row.of(value.getField(idCol), value.getField(inputCol));
            DenseIntDoubleVector[] hashValues = value.getFieldAs(outputCol);
            for (int i = 0; i < hashValues.length; i += 1) {
                out.collect(Row.join(kept, Row.of(i, hashValues[i])));
            }
        }
    }

    private static class IndexHashValueKeySelector
            implements KeySelector<Row, Tuple2<Integer, DenseIntDoubleVector>> {

        @Override
        public Tuple2<Integer, DenseIntDoubleVector> getKey(Row value) throws Exception {
            return Tuple2.of(value.getFieldAs(2), value.getFieldAs(3));
        }
    }

    private static class FilterByDistanceFunction extends RichFlatMapFunction<Row, Row> {
        private final double threshold;
        private LSHModelData modelData;

        public FilterByDistanceFunction(double threshold) {
            this.threshold = threshold;
        }

        @Override
        public void flatMap(Row value, Collector<Row> out) throws Exception {
            if (null == modelData) {
                modelData =
                        (LSHModelData)
                                getRuntimeContext().getBroadcastVariable(MODEL_DATA_BC_KEY).get(0);
            }
            double dist = modelData.keyDistance(value.getFieldAs(2), value.getFieldAs(3));
            if (dist <= threshold) {
                out.collect(Row.of(value.getFieldAs(0), value.getFieldAs(1), dist));
            }
        }
    }
}
