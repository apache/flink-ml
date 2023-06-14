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

package org.apache.flink.ml.clustering;

import org.apache.flink.api.common.time.Time;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.clustering.agglomerativeclustering.AgglomerativeClustering;
import org.apache.flink.ml.clustering.agglomerativeclustering.AgglomerativeClusteringParams;
import org.apache.flink.ml.common.distance.CosineDistanceMeasure;
import org.apache.flink.ml.common.distance.EuclideanDistanceMeasure;
import org.apache.flink.ml.common.distance.ManhattanDistanceMeasure;
import org.apache.flink.ml.common.window.CountTumblingWindows;
import org.apache.flink.ml.common.window.EventTimeTumblingWindows;
import org.apache.flink.ml.common.window.GlobalWindows;
import org.apache.flink.ml.common.window.ProcessingTimeTumblingWindows;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.time.Instant;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/** Tests {@link AgglomerativeClustering}. */
public class AgglomerativeClusteringTest extends AbstractTestBase {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamTableEnvironment tEnv;
    private StreamExecutionEnvironment env;
    private Table inputDataTable;

    private static final List<DenseIntDoubleVector> INPUT_DATA =
            Arrays.asList(
                    Vectors.dense(1, 1),
                    Vectors.dense(1, 4),
                    Vectors.dense(1, 0),
                    Vectors.dense(4, 4),
                    Vectors.dense(4, 1.5),
                    Vectors.dense(4, 0));

    private static final double[] EUCLIDEAN_AVERAGE_MERGE_DISTANCES =
            new double[] {1, 1.5, 3, 3.1394402, 3.9559706};

    private static final double[] COSINE_AVERAGE_MERGE_DISTANCES =
            new double[] {0, 1.1102230E-16, 0.0636708, 0.1425070, 0.3664484};

    private static final double[] MANHATTAN_AVERAGE_MERGE_DISTANCES =
            new double[] {1, 1.5, 3, 3.75, 4.875};
    private static final double[] EUCLIDEAN_SINGLE_MERGE_DISTANCES =
            new double[] {1, 1.5, 2.5, 3, 3};

    private static final double[] EUCLIDEAN_WARD_MERGE_DISTANCES =
            new double[] {1, 1.5, 3, 4.2573465, 5.5113519};

    private static final double[] EUCLIDEAN_COMPLETE_MERGE_DISTANCES =
            new double[] {1, 1.5, 3, 3.3541019, 5};

    private static final List<Set<DenseIntDoubleVector>> EUCLIDEAN_WARD_NUM_CLUSTERS_AS_TWO_RESULT =
            Arrays.asList(
                    new HashSet<>(
                            Arrays.asList(
                                    Vectors.dense(1, 1),
                                    Vectors.dense(1, 0),
                                    Vectors.dense(4, 1.5),
                                    Vectors.dense(4, 0))),
                    new HashSet<>(Arrays.asList(Vectors.dense(1, 4), Vectors.dense(4, 4))));

    private static final List<Set<DenseIntDoubleVector>> EUCLIDEAN_WARD_THRESHOLD_AS_TWO_RESULT =
            Arrays.asList(
                    new HashSet<>(Arrays.asList(Vectors.dense(1, 1), Vectors.dense(1, 0))),
                    new HashSet<>(Collections.singletonList(Vectors.dense(1, 4))),
                    new HashSet<>(Collections.singletonList(Vectors.dense(4, 4))),
                    new HashSet<>(Arrays.asList(Vectors.dense(4, 1.5), Vectors.dense(4, 0))));

    private static final List<Set<DenseIntDoubleVector>>
            EUCLIDEAN_WARD_COUNT_FIVE_WINDOW_AS_TWO_RESULT =
                    Arrays.asList(
                            new HashSet<>(Arrays.asList(Vectors.dense(1, 1), Vectors.dense(1, 0))),
                            new HashSet<>(
                                    Arrays.asList(
                                            Vectors.dense(1, 4),
                                            Vectors.dense(4, 4),
                                            Vectors.dense(4, 1.5))));

    private static final List<Set<DenseIntDoubleVector>>
            EUCLIDEAN_WARD_EVENT_TIME_WINDOW_AS_TWO_RESULT =
                    Arrays.asList(
                            new HashSet<>(Arrays.asList(Vectors.dense(1, 1), Vectors.dense(1, 0))),
                            new HashSet<>(Collections.singletonList(Vectors.dense(1, 4))),
                            new HashSet<>(
                                    Arrays.asList(Vectors.dense(4, 0), Vectors.dense(4, 1.5))),
                            new HashSet<>(Collections.singletonList(Vectors.dense(4, 4))));

    private static final List<Set<DenseIntDoubleVector>>
            EUCLIDEAN_AVERAGE_NUM_CLUSTERS_AS_TWO_RESULT =
                    Arrays.asList(
                            new HashSet<>(
                                    Arrays.asList(
                                            Vectors.dense(1, 1),
                                            Vectors.dense(1, 0),
                                            Vectors.dense(4, 1.5),
                                            Vectors.dense(4, 0))),
                            new HashSet<>(Arrays.asList(Vectors.dense(1, 4), Vectors.dense(4, 4))));

    private static final double TOLERANCE = 1e-7;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        inputDataTable =
                tEnv.fromDataStream(env.fromCollection(INPUT_DATA).map(x -> x)).as("features");
    }

    @Test
    public void testParam() {
        AgglomerativeClustering agglomerativeClustering = new AgglomerativeClustering();
        assertEquals("features", agglomerativeClustering.getFeaturesCol());
        assertEquals(2, agglomerativeClustering.getNumClusters().intValue());
        assertNull(agglomerativeClustering.getDistanceThreshold());
        assertEquals(AgglomerativeClustering.LINKAGE_WARD, agglomerativeClustering.getLinkage());
        assertEquals(EuclideanDistanceMeasure.NAME, agglomerativeClustering.getDistanceMeasure());
        assertFalse(agglomerativeClustering.getComputeFullTree());
        assertEquals("prediction", agglomerativeClustering.getPredictionCol());
        assertEquals(GlobalWindows.getInstance(), agglomerativeClustering.getWindows());

        agglomerativeClustering
                .setFeaturesCol("test_features")
                .setNumClusters(null)
                .setDistanceThreshold(0.01)
                .setLinkage(AgglomerativeClusteringParams.LINKAGE_AVERAGE)
                .setDistanceMeasure(CosineDistanceMeasure.NAME)
                .setComputeFullTree(true)
                .setPredictionCol("cluster_id")
                .setWindows(ProcessingTimeTumblingWindows.of(Time.milliseconds(100)));

        assertEquals("test_features", agglomerativeClustering.getFeaturesCol());
        assertNull(agglomerativeClustering.getNumClusters());
        assertEquals(0.01, agglomerativeClustering.getDistanceThreshold(), TOLERANCE);
        assertEquals(AgglomerativeClustering.LINKAGE_AVERAGE, agglomerativeClustering.getLinkage());
        assertEquals(CosineDistanceMeasure.NAME, agglomerativeClustering.getDistanceMeasure());
        assertTrue(agglomerativeClustering.getComputeFullTree());
        assertEquals("cluster_id", agglomerativeClustering.getPredictionCol());
        assertEquals(
                ProcessingTimeTumblingWindows.of(Time.milliseconds(100)),
                agglomerativeClustering.getWindows());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable =
                tEnv.fromDataStream(env.fromElements(Row.of("", "")))
                        .as("test_features", "dummy_input");
        AgglomerativeClustering agglomerativeClustering =
                new AgglomerativeClustering()
                        .setFeaturesCol("test_features")
                        .setPredictionCol("test_prediction");
        Table[] outputs = agglomerativeClustering.transform(tempTable);
        assertEquals(2, outputs.length);
        assertEquals(
                Arrays.asList("test_features", "dummy_input", "test_prediction"),
                outputs[0].getResolvedSchema().getColumnNames());
        assertEquals(
                Arrays.asList("clusterId1", "clusterId2", "distance", "sizeOfMergedCluster"),
                outputs[1].getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() throws Exception {
        Table[] outputs;
        AgglomerativeClustering agglomerativeClustering =
                new AgglomerativeClustering()
                        .setLinkage(AgglomerativeClusteringParams.LINKAGE_WARD)
                        .setDistanceMeasure(EuclideanDistanceMeasure.NAME)
                        .setPredictionCol("pred");

        // Tests euclidean distance with linkage as ward, numClusters = 2.
        outputs = agglomerativeClustering.transform(inputDataTable);
        verifyClusteringResult(
                EUCLIDEAN_WARD_NUM_CLUSTERS_AS_TWO_RESULT,
                outputs[0],
                agglomerativeClustering.getFeaturesCol(),
                agglomerativeClustering.getPredictionCol());

        // Tests euclidean distance with linkage as ward, numClusters = 2, compute_full_tree =
        // true.
        outputs = agglomerativeClustering.setComputeFullTree(true).transform(inputDataTable);
        verifyClusteringResult(
                EUCLIDEAN_WARD_NUM_CLUSTERS_AS_TWO_RESULT,
                outputs[0],
                agglomerativeClustering.getFeaturesCol(),
                agglomerativeClustering.getPredictionCol());

        // Tests euclidean distance with linkage as ward, distance_threshold = 2.
        outputs =
                agglomerativeClustering
                        .setNumClusters(null)
                        .setDistanceThreshold(2.0)
                        .transform(inputDataTable);
        verifyClusteringResult(
                EUCLIDEAN_WARD_THRESHOLD_AS_TWO_RESULT,
                outputs[0],
                agglomerativeClustering.getFeaturesCol(),
                agglomerativeClustering.getPredictionCol());
    }

    @Test
    public void testTransformWithAverageLinkage() throws Exception {
        AgglomerativeClustering agglomerativeClustering =
                new AgglomerativeClustering()
                        .setLinkage(AgglomerativeClusteringParams.LINKAGE_AVERAGE)
                        .setDistanceMeasure(EuclideanDistanceMeasure.NAME)
                        .setNumClusters(2)
                        .setPredictionCol("pred");

        Table output = agglomerativeClustering.transform(inputDataTable)[0];
        verifyClusteringResult(
                EUCLIDEAN_AVERAGE_NUM_CLUSTERS_AS_TWO_RESULT,
                output,
                agglomerativeClustering.getFeaturesCol(),
                agglomerativeClustering.getPredictionCol());
    }

    @Test
    public void testLargeDistanceThreshold() throws Exception {
        AgglomerativeClustering agglomerativeClustering =
                new AgglomerativeClustering()
                        .setNumClusters(null)
                        .setDistanceThreshold(Double.MAX_VALUE);
        Table output = agglomerativeClustering.transform(inputDataTable)[0];
        HashSet<Integer> clusterIds = new HashSet<>();
        tEnv.toDataStream(output)
                .executeAndCollect()
                .forEachRemaining(
                        x ->
                                clusterIds.add(
                                        x.getFieldAs(agglomerativeClustering.getPredictionCol())));
        assertEquals(1, clusterIds.size());
    }

    @Test
    public void testTransformWithCountTumblingWindows() throws Exception {
        env.setParallelism(1);

        inputDataTable =
                tEnv.fromDataStream(env.fromCollection(INPUT_DATA).map(x -> x)).as("features");

        AgglomerativeClustering agglomerativeClustering =
                new AgglomerativeClustering()
                        .setLinkage(AgglomerativeClusteringParams.LINKAGE_WARD)
                        .setDistanceMeasure(EuclideanDistanceMeasure.NAME)
                        .setPredictionCol("pred")
                        .setWindows(CountTumblingWindows.of(5));

        Table[] outputs = agglomerativeClustering.transform(inputDataTable);
        verifyClusteringResult(
                EUCLIDEAN_WARD_COUNT_FIVE_WINDOW_AS_TWO_RESULT,
                outputs[0],
                agglomerativeClustering.getFeaturesCol(),
                agglomerativeClustering.getPredictionCol());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testTransformWithEventTimeTumblingWindows() throws Exception {
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        new TypeInformation<?>[] {
                            DenseIntDoubleVectorTypeInfo.INSTANCE, Types.INSTANT
                        },
                        new String[] {"features", "ts"});

        Instant baseTime = Instant.now();
        DataStream<Row> inputDataStream =
                env.fromCollection(INPUT_DATA)
                        .setParallelism(1)
                        .map(x -> Row.of(x, baseTime.plusSeconds((long) x.get(0))), outputTypeInfo);

        Schema schema =
                Schema.newBuilder()
                        .column("features", DataTypes.of(DenseIntDoubleVectorTypeInfo.INSTANCE))
                        .column("ts", DataTypes.TIMESTAMP_LTZ(3))
                        .watermark("ts", "ts - INTERVAL '5' SECOND")
                        .build();

        Table inputDataTable = tEnv.fromDataStream(inputDataStream, schema);

        AgglomerativeClustering agglomerativeClustering =
                new AgglomerativeClustering()
                        .setLinkage(AgglomerativeClusteringParams.LINKAGE_WARD)
                        .setDistanceMeasure(EuclideanDistanceMeasure.NAME)
                        .setPredictionCol("pred")
                        .setWindows(EventTimeTumblingWindows.of(Time.seconds(1)));

        Table[] outputs = agglomerativeClustering.transform(inputDataTable);

        List<Row> output = IteratorUtils.toList(tEnv.toDataStream(outputs[0]).executeAndCollect());
        List<Set<DenseIntDoubleVector>> actualGroups =
                KMeansTest.groupFeaturesByPrediction(
                        output,
                        agglomerativeClustering.getFeaturesCol(),
                        agglomerativeClustering.getPredictionCol());

        boolean isAllSubSet = true;
        for (Set<DenseIntDoubleVector> expectedSet :
                EUCLIDEAN_WARD_EVENT_TIME_WINDOW_AS_TWO_RESULT) {
            boolean isSubset = false;
            for (Set<DenseIntDoubleVector> actualSet : actualGroups) {
                if (actualSet.containsAll(expectedSet)) {
                    isSubset = true;
                    break;
                }
            }
            isAllSubSet &= isSubset;
        }
        assertTrue(isAllSubSet);
    }

    @Test
    public void testMergeInfo() throws Exception {
        Table[] outputs;
        AgglomerativeClustering agglomerativeClustering =
                new AgglomerativeClustering()
                        .setLinkage(AgglomerativeClusteringParams.LINKAGE_AVERAGE)
                        .setDistanceMeasure(EuclideanDistanceMeasure.NAME)
                        .setPredictionCol("pred")
                        .setComputeFullTree(true);

        // Tests euclidean distance with linkage as average.
        outputs = agglomerativeClustering.transform(inputDataTable);
        verifyMergeInfo(EUCLIDEAN_AVERAGE_MERGE_DISTANCES, outputs[1]);

        // Tests cosine distance with linkage as average.
        agglomerativeClustering.setDistanceMeasure(CosineDistanceMeasure.NAME);
        outputs = agglomerativeClustering.transform(inputDataTable);
        verifyMergeInfo(COSINE_AVERAGE_MERGE_DISTANCES, outputs[1]);

        // Tests manhattan distance with linkage as average.
        agglomerativeClustering.setDistanceMeasure(ManhattanDistanceMeasure.NAME);
        outputs = agglomerativeClustering.transform(inputDataTable);
        verifyMergeInfo(MANHATTAN_AVERAGE_MERGE_DISTANCES, outputs[1]);

        // Tests euclidean distance with linkage as complete.
        agglomerativeClustering
                .setDistanceMeasure(EuclideanDistanceMeasure.NAME)
                .setLinkage(AgglomerativeClusteringParams.LINKAGE_COMPLETE);
        outputs = agglomerativeClustering.transform(inputDataTable);
        verifyMergeInfo(EUCLIDEAN_COMPLETE_MERGE_DISTANCES, outputs[1]);

        // Tests euclidean distance with linkage as single.
        agglomerativeClustering.setLinkage(AgglomerativeClusteringParams.LINKAGE_SINGLE);
        outputs = agglomerativeClustering.transform(inputDataTable);
        verifyMergeInfo(EUCLIDEAN_SINGLE_MERGE_DISTANCES, outputs[1]);

        // Tests euclidean distance with linkage as ward.
        agglomerativeClustering.setLinkage(AgglomerativeClusteringParams.LINKAGE_WARD);
        outputs = agglomerativeClustering.transform(inputDataTable);
        verifyMergeInfo(EUCLIDEAN_WARD_MERGE_DISTANCES, outputs[1]);

        // Tests merge info not fully computed.
        agglomerativeClustering.setComputeFullTree(false);
        outputs = agglomerativeClustering.transform(inputDataTable);
        verifyMergeInfo(
                Arrays.copyOfRange(
                        EUCLIDEAN_WARD_MERGE_DISTANCES,
                        0,
                        EUCLIDEAN_WARD_MERGE_DISTANCES.length - 1),
                outputs[1]);
    }

    @Test
    public void testSaveLoadTransform() throws Exception {
        AgglomerativeClustering agglomerativeClustering =
                new AgglomerativeClustering()
                        .setLinkage(AgglomerativeClusteringParams.LINKAGE_WARD)
                        .setDistanceMeasure(EuclideanDistanceMeasure.NAME)
                        .setPredictionCol("pred");

        agglomerativeClustering =
                TestUtils.saveAndReload(
                        tEnv,
                        agglomerativeClustering,
                        tempFolder.newFolder().getAbsolutePath(),
                        AgglomerativeClustering::load);

        Table[] outputs = agglomerativeClustering.transform(inputDataTable);
        verifyClusteringResult(
                EUCLIDEAN_WARD_NUM_CLUSTERS_AS_TWO_RESULT,
                outputs[0],
                agglomerativeClustering.getFeaturesCol(),
                agglomerativeClustering.getPredictionCol());
    }

    @SuppressWarnings("unchecked")
    private void verifyMergeInfo(double[] expectedDistances, Table mergeInfoTable)
            throws Exception {
        List<Row> mergeInfo =
                IteratorUtils.toList(tEnv.toDataStream(mergeInfoTable).executeAndCollect());
        assertEquals(expectedDistances.length, mergeInfo.size());
        for (int i = 0; i < mergeInfo.size(); i++) {
            double actualDistance = ((Number) mergeInfo.get(i).getFieldAs(2)).doubleValue();
            assertEquals(expectedDistances[i], actualDistance, TOLERANCE);
        }
    }

    @SuppressWarnings("unchecked")
    public void verifyClusteringResult(
            List<Set<DenseIntDoubleVector>> expected,
            Table outputTable,
            String featureCol,
            String predictionCol)
            throws Exception {
        List<Row> output = IteratorUtils.toList(tEnv.toDataStream(outputTable).executeAndCollect());
        List<Set<DenseIntDoubleVector>> actualGroups =
                KMeansTest.groupFeaturesByPrediction(output, featureCol, predictionCol);
        assertTrue(CollectionUtils.isEqualCollection(expected, actualGroups));
    }
}
