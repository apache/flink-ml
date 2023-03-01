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

package org.apache.flink.ml.common.gbt;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.common.gbt.defs.BoostingStrategy;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.test.util.TestBaseUtils;
import org.apache.flink.testutils.junit.SharedObjects;
import org.apache.flink.testutils.junit.SharedReference;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;

/** Tests {@link Preprocess}. */
public class PreprocessTest extends AbstractTestBase {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    @Rule public final SharedObjects sharedObjects = SharedObjects.create();

    private static final List<Row> inputDataRows =
            Arrays.asList(
                    Row.of(1.2, 2, null, 40., Vectors.dense(1.2, 2, Double.NaN)),
                    Row.of(2.3, 3, "b", 40., Vectors.dense(2.3, 3, 2.)),
                    Row.of(3.4, 4, "c", 40., Vectors.dense(3.4, 4, 3.)),
                    Row.of(4.5, 5, "a", 40., Vectors.dense(4.5, 5, 1.)),
                    Row.of(5.6, 2, "b", 40., Vectors.dense(5.6, 2, 2.)),
                    Row.of(null, 3, "c", 41., Vectors.dense(Double.NaN, 3, 3.)),
                    Row.of(12.8, 4, "e", 41., Vectors.dense(12.8, 4, 5.)),
                    Row.of(13.9, 2, "b", 41., Vectors.dense(13.9, 2, 2.)),
                    Row.of(14.1, 4, "a", 41., Vectors.dense(14.1, 4, 1.)),
                    Row.of(15.3, 1, "d", 41., Vectors.dense(15.3, 1, 4.)));

    private StreamTableEnvironment tEnv;
    private Table inputTable;
    private SharedReference<ArrayBlockingQueue<FeatureMeta>> actualMeta;

    //    private static void verifyPredictionResult(Table output, List<Row> expected) throws
    // Exception {
    //        StreamTableEnvironment tEnv =
    //                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
    //        //noinspection unchecked
    //        List<Row> results =
    // IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
    //        final double delta = 1e-3;
    //        final Comparator<DenseVector> denseVectorComparator =
    //                new TestUtils.DenseVectorComparatorWithDelta(delta);
    //        final Comparator<Row> comparator =
    //                Comparator.<Row, Double>comparing(d -> d.getFieldAs(0))
    //                        .thenComparing(d -> d.getFieldAs(1), denseVectorComparator)
    //                        .thenComparing(d -> d.getFieldAs(2), denseVectorComparator);
    //        TestUtils.compareResultCollectionsWithComparator(expected, results, comparator);
    //    }

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        inputTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                inputDataRows,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            Types.DOUBLE,
                                            Types.INT,
                                            Types.STRING,
                                            Types.DOUBLE,
                                            DenseVectorTypeInfo.INSTANCE
                                        },
                                        new String[] {"f0", "f1", "f2", "label", "vec"})));

        actualMeta = sharedObjects.add(new ArrayBlockingQueue<>(8));
    }

    private static class CollectSink<T> implements SinkFunction<T> {
        private final SharedReference<ArrayBlockingQueue<T>> q;

        public CollectSink(SharedReference<ArrayBlockingQueue<T>> q) {
            this.q = q;
        }

        @Override
        public void invoke(T value, Context context) {
            q.get().add(value);
        }
    }

    @Test
    public void testPreprocessCols() throws Exception {
        BoostingStrategy strategy = new BoostingStrategy();
        strategy.isInputVector = false;
        strategy.featuresCols = new String[] {"f0", "f1", "f2"};
        strategy.categoricalCols = new String[] {"f2"};
        strategy.labelCol = "label";
        strategy.maxBins = 3;
        Tuple2<Table, DataStream<FeatureMeta>> results =
                Preprocess.preprocessCols(inputTable, strategy);

        actualMeta.get().clear();
        results.f1.addSink(new CollectSink<>(actualMeta));
        //noinspection unchecked
        List<Row> preprocessedRows =
                IteratorUtils.toList(tEnv.toDataStream(results.f0).executeAndCollect());

        // TODO: correct `binEdges` of feature `f0` after FLINK-30734 resolved.
        List<FeatureMeta> expectedMeta =
                Arrays.asList(
                        FeatureMeta.continuous("f0", 3, new double[] {1.2, 4.5, 13.9, 15.3}),
                        FeatureMeta.continuous("f1", 3, new double[] {1.0, 2.0, 4.0, 5.0}),
                        FeatureMeta.categorical("f2", 5, new String[] {"a", "b", "c", "d", "e"}));

        List<Row> expectedPreprocessedRows =
                Arrays.asList(
                        Row.of(40.0, 0, 1, 5.0),
                        Row.of(40.0, 0, 1, 1.0),
                        Row.of(40.0, 0, 2, 2.0),
                        Row.of(40.0, 1, 2, 0.0),
                        Row.of(40.0, 1, 1, 1.0),
                        Row.of(41.0, 3, 1, 2.0),
                        Row.of(41.0, 1, 2, 4.0),
                        Row.of(41.0, 2, 1, 1.0),
                        Row.of(41.0, 2, 2, 0.0),
                        Row.of(41.0, 2, 0, 3.0));
        Comparator<Row> preprocessedRowComparator =
                Comparator.<Row, Double>comparing(d -> d.getFieldAs(0))
                        .thenComparing(d -> d.getFieldAs(1))
                        .thenComparing(d -> d.getFieldAs(2))
                        .thenComparing(d -> d.getFieldAs(3));

        TestBaseUtils.compareResultCollections(
                expectedPreprocessedRows, preprocessedRows, preprocessedRowComparator);
        TestBaseUtils.compareResultCollections(
                expectedMeta, new ArrayList<>(actualMeta.get()), Comparator.comparing(d -> d.name));
    }

    @Test
    public void testPreprocessVectorCol() throws Exception {
        BoostingStrategy strategy = new BoostingStrategy();
        strategy.isInputVector = true;
        strategy.featuresCols = new String[] {"vec"};
        strategy.labelCol = "label";
        strategy.maxBins = 3;
        Tuple2<Table, DataStream<FeatureMeta>> results =
                Preprocess.preprocessVecCol(inputTable, strategy);

        actualMeta.get().clear();
        results.f1.addSink(new CollectSink<>(actualMeta));
        //noinspection unchecked
        List<Row> preprocessedRows =
                IteratorUtils.toList(tEnv.toDataStream(results.f0).executeAndCollect());

        // TODO: correct `binEdges` of feature `_vec_f0` and `_vec_f2` after FLINK-30734 resolved.
        List<FeatureMeta> expectedMeta =
                Arrays.asList(
                        FeatureMeta.continuous("_vec_f0", 3, new double[] {1.2, 4.5, 13.9, 15.3}),
                        FeatureMeta.continuous("_vec_f1", 3, new double[] {1.0, 2.0, 4.0, 5.0}),
                        FeatureMeta.continuous("_vec_f2", 3, new double[] {1.0, 2.0, 3.0, 5.0}));
        List<Row> expectedPreprocessedRows =
                Arrays.asList(
                        Row.of(40.0, Vectors.dense(0, 1, 3.0)),
                        Row.of(40.0, Vectors.dense(0, 1, 1.0)),
                        Row.of(40.0, Vectors.dense(0, 2, 2.0)),
                        Row.of(40.0, Vectors.dense(1, 2, 0.0)),
                        Row.of(40.0, Vectors.dense(1, 1, 1.0)),
                        Row.of(41.0, Vectors.dense(3, 1, 2.0)),
                        Row.of(41.0, Vectors.dense(1, 2, 2.0)),
                        Row.of(41.0, Vectors.dense(2, 1, 1.0)),
                        Row.of(41.0, Vectors.dense(2, 2, 0.0)),
                        Row.of(41.0, Vectors.dense(2, 0, 2.0)));

        Comparator<Row> preprocessedRowComparator =
                Comparator.<Row, Double>comparing(d -> d.getFieldAs(0))
                        .thenComparing(d -> d.getFieldAs(1), TestUtils::compare);

        TestBaseUtils.compareResultCollections(
                expectedPreprocessedRows, preprocessedRows, preprocessedRowComparator);
        TestBaseUtils.compareResultCollections(
                expectedMeta, new ArrayList<>(actualMeta.get()), Comparator.comparing(d -> d.name));
    }
}
