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

package org.apache.flink.ml.feature;

import org.apache.flink.ml.feature.lsh.MinHashLSH;
import org.apache.flink.ml.feature.lsh.MinHashLSHModel;
import org.apache.flink.ml.feature.lsh.MinHashLSHModelData;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static org.apache.flink.table.api.Expressions.$;

/** Tests {@link MinHashLSH} and {@link MinHashLSHModel}. */
public class MinHashLSHTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private final List<Row> outputRows =
            convertToOutputFormat(
                    Arrays.asList(
                            new double[][] {
                                {1.73046954E8, 1.57275425E8, 6.90717571E8},
                                {5.02301169E8, 7.967141E8, 4.06089319E8},
                                {2.83652171E8, 1.97714719E8, 6.04731316E8},
                                {5.2181506E8, 6.36933726E8, 6.13894128E8},
                                {3.04301769E8, 1.113672955E9, 6.1388711E8}
                            },
                            new double[][] {
                                {1.73046954E8, 1.57275425E8, 6.7798584E7},
                                {6.38582806E8, 1.78703694E8, 4.06089319E8},
                                {6.232638E8, 9.28867E7, 9.92010642E8},
                                {2.461064E8, 1.12787481E8, 1.92180297E8},
                                {2.38162496E8, 1.552933319E9, 2.77995137E8}
                            },
                            new double[][] {
                                {1.73046954E8, 1.57275425E8, 6.90717571E8},
                                {1.453197722E9, 7.967141E8, 4.06089319E8},
                                {6.232638E8, 1.97714719E8, 6.04731316E8},
                                {2.461064E8, 1.12787481E8, 1.92180297E8},
                                {1.224130231E9, 1.113672955E9, 2.77995137E8}
                            }));
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table inputTable;

    /**
     * Converts a list of 2d double arrays to a list of rows with each of which containing a
     * DenseVector array.
     */
    private static List<Row> convertToOutputFormat(List<double[][]> arrays) {
        return arrays.stream()
                .map(
                        array -> {
                            DenseVector[] denseVectors =
                                    Arrays.stream(array)
                                            .map(Vectors::dense)
                                            .toArray(DenseVector[]::new);
                            return Row.of((Object) denseVectors);
                        })
                .collect(Collectors.toList());
    }

    private static void verifyPredictionResult(Table output, List<Row> expected) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        List<Row> results = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        compareResultCollections(
                expected,
                results,
                (d0, d1) -> {
                    DenseVectorArrayComparator denseVectorArrayComparator =
                            new DenseVectorArrayComparator();
                    return denseVectorArrayComparator.compare(d0.getFieldAs(0), d1.getFieldAs(0));
                });
    }

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        List<Row> inputRows =
                Arrays.asList(
                        Row.of(
                                0,
                                Vectors.sparse(6, new int[] {0, 1, 2}, new double[] {1., 1., 1.})),
                        Row.of(
                                1,
                                Vectors.sparse(6, new int[] {2, 3, 4}, new double[] {1., 1., 1.})),
                        Row.of(
                                2,
                                Vectors.sparse(6, new int[] {0, 2, 4}, new double[] {1., 1., 1.})));

        Schema schema =
                Schema.newBuilder()
                        .column("f0", DataTypes.INT())
                        .column("f1", DataTypes.of(SparseVector.class))
                        .build();
        DataStream<Row> dataStream = env.fromCollection(inputRows);

        inputTable = tEnv.fromDataStream(dataStream, schema).as("id", "vec");
    }

    @Test
    public void testHashFunction() {
        MinHashLSHModelData lshModelData =
                new MinHashLSHModelData(3, 1, new int[] {0, 1, 3}, new int[] {1, 2, 0});
        Vector vec = Vectors.sparse(10, new int[] {2, 3, 5, 7}, new double[] {1., 1., 1., 1.});
        DenseVector[] result = lshModelData.hashFunction(vec);
        Assert.assertEquals(3, result.length);
        Assert.assertEquals(Vectors.dense(1.), result[0]);
        Assert.assertEquals(Vectors.dense(5.), result[1]);
        Assert.assertEquals(Vectors.dense(9.), result[2]);
    }

    @Test
    public void testHashFunctionEqualWithSparseDenseVector() {
        // Uses randomly generate coefficients, so that the hash values are not always from the
        // least non-zero index.
        MinHashLSHModelData lshModelData = MinHashLSHModelData.generateModelData(3, 1, 10, 2022L);
        new MinHashLSHModelData(3, 1, new int[] {0, 1, 3}, new int[] {1, 2, 0});
        Vector vec = Vectors.sparse(10, new int[] {2, 3, 5, 7}, new double[] {1., 1., 1., 1.});
        DenseVector[] denseResults = lshModelData.hashFunction(vec.toDense());
        DenseVector[] sparseResults = lshModelData.hashFunction(vec.toSparse());
        Assert.assertArrayEquals(denseResults, sparseResults);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testHashFunctionWithEmptyVector() {
        MinHashLSHModelData lshModelData =
                new MinHashLSHModelData(3, 1, new int[] {0, 1, 3}, new int[] {1, 2, 0});
        Vector vec = Vectors.sparse(10, new int[] {}, new double[] {});
        lshModelData.hashFunction(vec);
    }

    @Test
    public void testParam() {
        MinHashLSH lsh = new MinHashLSH();
        Assert.assertEquals("input", lsh.getInputCol());
        Assert.assertEquals("output", lsh.getOutputCol());
        Assert.assertEquals(MinHashLSH.class.getName().hashCode(), lsh.getSeed());
        Assert.assertEquals(1, lsh.getNumHashTables());
        Assert.assertEquals(1, lsh.getNumHashFunctionsPerTable());
        lsh.setInputCol("vec")
                .setOutputCol("hashes")
                .setSeed(2022L)
                .setNumHashTables(3)
                .setNumHashFunctionsPerTable(4);
        Assert.assertEquals("vec", lsh.getInputCol());
        Assert.assertEquals("hashes", lsh.getOutputCol());
        Assert.assertEquals(2022L, lsh.getSeed());
        Assert.assertEquals(3, lsh.getNumHashTables());
        Assert.assertEquals(4, lsh.getNumHashFunctionsPerTable());
    }

    @Test
    public void testOutputSchema() throws Exception {
        MinHashLSH lsh =
                new MinHashLSH()
                        .setInputCol("vec")
                        .setOutputCol("hashes")
                        .setSeed(2022L)
                        .setNumHashTables(5)
                        .setNumHashFunctionsPerTable(3);
        MinHashLSHModel model = lsh.fit(inputTable);
        Table output = model.transform(inputTable)[0];
        Assert.assertEquals(
                Arrays.asList("id", "vec", "hashes"), output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        MinHashLSH lsh =
                new MinHashLSH()
                        .setInputCol("vec")
                        .setOutputCol("hashes")
                        .setSeed(2022L)
                        .setNumHashTables(5)
                        .setNumHashFunctionsPerTable(3);
        MinHashLSHModel lshModel = lsh.fit(inputTable);
        Table output = lshModel.transform(inputTable)[0].select($(lsh.getOutputCol()));
        verifyPredictionResult(output, outputRows);
    }

    @Test
    public void testFitAndPredictWithNumHashFunctionPerTableIsOne() throws Exception {
        // When numHashFunctionPerTable = 1 and other parameters are same, the results should be the
        // same with SparkML.
        final List<Row> expected =
                convertToOutputFormat(
                        Arrays.asList(
                                new double[][] {
                                    {1.73046954E8},
                                    {1.57275425E8},
                                    {6.7798584E7},
                                    {6.38582806E8},
                                    {1.78703694E8}
                                },
                                new double[][] {
                                    {1.73046954E8},
                                    {1.57275425E8},
                                    {6.90717571E8},
                                    {5.02301169E8},
                                    {7.967141E8}
                                },
                                new double[][] {
                                    {1.73046954E8},
                                    {1.57275425E8},
                                    {6.90717571E8},
                                    {1.453197722E9},
                                    {7.967141E8}
                                }));
        MinHashLSH lsh =
                new MinHashLSH()
                        .setInputCol("vec")
                        .setOutputCol("hashes")
                        .setSeed(2022L)
                        .setNumHashTables(5);
        MinHashLSHModel lshModel = lsh.fit(inputTable);
        Table output = lshModel.transform(inputTable)[0].select($(lsh.getOutputCol()));
        verifyPredictionResult(output, expected);
    }

    @Test
    public void testEstimatorSaveLoadAndPredict() throws Exception {
        MinHashLSH lsh =
                new MinHashLSH()
                        .setInputCol("vec")
                        .setOutputCol("hashes")
                        .setSeed(2022L)
                        .setNumHashTables(5)
                        .setNumHashFunctionsPerTable(3);

        MinHashLSH loadedLsh =
                TestUtils.saveAndReload(
                        tEnv, lsh, tempFolder.newFolder().getAbsolutePath(), MinHashLSH::load);
        MinHashLSHModel lshModel = loadedLsh.fit(inputTable);
        Assert.assertEquals(
                Arrays.asList(
                        "numHashTables",
                        "numHashFunctionsPerTable",
                        "randCoefficientA",
                        "randCoefficientB"),
                lshModel.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = lshModel.transform(inputTable)[0].select($(lsh.getOutputCol()));
        verifyPredictionResult(output, outputRows);
    }

    @Test
    public void testModelSaveLoadAndPredict() throws Exception {
        MinHashLSH lsh =
                new MinHashLSH()
                        .setInputCol("vec")
                        .setOutputCol("hashes")
                        .setSeed(2022L)
                        .setNumHashTables(5)
                        .setNumHashFunctionsPerTable(3);
        MinHashLSHModel lshModel = lsh.fit(inputTable);
        MinHashLSHModel loadedModel =
                TestUtils.saveAndReload(
                        tEnv,
                        lshModel,
                        tempFolder.newFolder().getAbsolutePath(),
                        MinHashLSHModel::load);
        Table output = loadedModel.transform(inputTable)[0].select($(lsh.getOutputCol()));
        verifyPredictionResult(output, outputRows);
    }

    @Test
    public void testGetModelData() throws Exception {
        MinHashLSH lsh =
                new MinHashLSH()
                        .setInputCol("vec")
                        .setOutputCol("hashes")
                        .setSeed(2022L)
                        .setNumHashTables(5)
                        .setNumHashFunctionsPerTable(3);

        MinHashLSHModel lshModel = lsh.fit(inputTable);
        Table modelDataTable = lshModel.getModelData()[0];
        List<String> modelDataColumnNames = modelDataTable.getResolvedSchema().getColumnNames();
        DataStream<Row> output = tEnv.toDataStream(modelDataTable);
        Assert.assertArrayEquals(
                new String[] {
                    "numHashTables",
                    "numHashFunctionsPerTable",
                    "randCoefficientA",
                    "randCoefficientB"
                },
                modelDataColumnNames.toArray(new String[0]));

        Row modelDataRow = (Row) IteratorUtils.toList(output.executeAndCollect()).get(0);
        MinHashLSHModelData modelData =
                new MinHashLSHModelData(
                        modelDataRow.getFieldAs(0),
                        modelDataRow.getFieldAs(1),
                        modelDataRow.getFieldAs(2),
                        modelDataRow.getFieldAs(3));
        Assert.assertNotNull(modelData);
        Assert.assertEquals(lsh.getNumHashTables(), modelData.numHashTables);
        Assert.assertEquals(lsh.getNumHashFunctionsPerTable(), modelData.numHashFunctionsPerTable);
        Assert.assertEquals(
                lsh.getNumHashTables() * lsh.getNumHashFunctionsPerTable(),
                modelData.randCoefficientA.length);
        Assert.assertEquals(
                lsh.getNumHashTables() * lsh.getNumHashFunctionsPerTable(),
                modelData.randCoefficientB.length);
    }

    @Test
    public void testSetModelData() throws Exception {
        MinHashLSH lsh =
                new MinHashLSH()
                        .setInputCol("vec")
                        .setOutputCol("hashes")
                        .setSeed(2022L)
                        .setNumHashTables(5)
                        .setNumHashFunctionsPerTable(3);
        MinHashLSHModel modelA = lsh.fit(inputTable);
        Table modelDataData = modelA.getModelData()[0];
        MinHashLSHModel modelB = new MinHashLSHModel().setModelData(modelDataData);
        ParamUtils.updateExistingParams(modelB, modelA.getParamMap());
        Table output = modelB.transform(inputTable)[0].select($(lsh.getOutputCol()));
        verifyPredictionResult(output, outputRows);
    }

    @Test
    public void testApproxNearestNeighbors() {
        MinHashLSH lsh =
                new MinHashLSH()
                        .setInputCol("vec")
                        .setOutputCol("hashes")
                        .setSeed(2022L)
                        .setNumHashTables(5)
                        .setNumHashFunctionsPerTable(1);
        MinHashLSHModel lshModel = lsh.fit(inputTable);
        List<Row> expected = Arrays.asList(Row.of(0, .75), Row.of(1, .75));

        Vector key = Vectors.sparse(6, new int[] {1, 3}, new double[] {1.0, 1.0});
        Table output =
                lshModel.approxNearestNeighbors(inputTable, key, 2).select($("id"), $("distCol"));
        List<Row> results = IteratorUtils.toList(output.execute().collect());
        compareResultCollections(expected, results, Comparator.comparing(r -> r.getFieldAs(0)));
    }

    @Test
    public void testApproxSimilarityJoin() {
        MinHashLSH lsh =
                new MinHashLSH()
                        .setInputCol("vec")
                        .setOutputCol("hashes")
                        .setSeed(2022L)
                        .setNumHashTables(5)
                        .setNumHashFunctionsPerTable(1);
        Table dataA = inputTable;
        MinHashLSHModel lshModel = lsh.fit(dataA);

        List<Row> inputRowsB =
                Arrays.asList(
                        Row.of(
                                3,
                                Vectors.sparse(6, new int[] {1, 3, 5}, new double[] {1., 1., 1.})),
                        Row.of(
                                4,
                                Vectors.sparse(6, new int[] {2, 3, 5}, new double[] {1., 1., 1.})),
                        Row.of(
                                5,
                                Vectors.sparse(6, new int[] {1, 2, 4}, new double[] {1., 1., 1.})));
        Schema schema =
                Schema.newBuilder()
                        .column("f0", DataTypes.INT())
                        .column("f1", DataTypes.of(SparseVector.class))
                        .build();
        Table dataB = tEnv.fromDataStream(env.fromCollection(inputRowsB), schema).as("id", "vec");

        List<Row> expected =
                Arrays.asList(
                        Row.of(1, 4, 0.5), Row.of(0, 5, 0.5), Row.of(1, 5, 0.5), Row.of(2, 5, 0.5));

        Table output = lshModel.approxSimilarityJoin(dataA, dataB, .6, "id");
        List<Row> results = IteratorUtils.toList(output.execute().collect());
        compareResultCollections(
                expected,
                results,
                Comparator.<Row>comparingInt(r -> r.getFieldAs(0))
                        .thenComparingInt(r -> r.getFieldAs(1))
                        .thenComparingDouble(r -> r.getFieldAs(2)));
    }

    private static class DenseVectorArrayComparator implements Comparator<DenseVector[]> {
        @Override
        public int compare(DenseVector[] o1, DenseVector[] o2) {
            if (o1.length != o2.length) {
                return o1.length - o2.length;
            }
            for (int i = 0; i < o1.length; i += 1) {
                int cmp = TestUtils.compare(o1[i], o2[i]);
                if (0 != cmp) {
                    return cmp;
                }
            }
            return 0;
        }
    }
}
