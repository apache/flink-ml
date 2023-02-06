################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import os

from pyflink.common import Types
from pyflink.common.time import Time, Instant
from pyflink.java_gateway import get_gateway
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.common.window import GlobalWindows, EventTimeTumblingWindows, CountTumblingWindows
from pyflink.ml.clustering.agglomerativeclustering import AgglomerativeClustering
from pyflink.ml.clustering.tests.test_kmeans import group_features_by_prediction
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase
from pyflink.table import Schema
from pyflink.table.expressions import col


class AgglomerativeClusteringTest(PyFlinkMLTestCase):
    def setUp(self):
        super(AgglomerativeClusteringTest, self).setUp()
        self.input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([1, 1]),),
                (Vectors.dense([1, 4]),),
                (Vectors.dense([1, 0]),),
                (Vectors.dense([4, 1.5]),),
                (Vectors.dense([4, 4]),),
                (Vectors.dense([4, 0]),),
            ],
                type_info=Types.ROW_NAMED(
                    ['features'],
                    [DenseVectorTypeInfo()])))

        self.euclidean_average_merge_distances = [1.0, 1.5, 3.0, 3.1394402, 3.9559706]
        self.cosine_average_merge_distances = [0, 1.1102230E-16, 0.0636708, 0.1425070, 0.3664484]
        self.manhattan_average_merge_distances = [1, 1.5, 3, 3.75, 4.875]
        self.euclidean_single_merge_distances = [1, 1.5, 2.5, 3, 3]
        self.euclidean_ward_merge_distances = [1, 1.5, 3, 4.2573465, 5.5113519]
        self.euclidean_complete_merge_distances = [1, 1.5, 3, 3.3541019, 5]

        self.euclidean_ward_num_clusters_as_two_result = [
            {Vectors.dense(1, 1), Vectors.dense(1, 0), Vectors.dense(4, 1.5), Vectors.dense(4, 0)},
            {Vectors.dense(1, 4), Vectors.dense(4, 4)}
        ]

        self.euclidean_ward_threshold_as_two_result = [
            {Vectors.dense(1, 1), Vectors.dense(1, 0)},
            {Vectors.dense(1, 4)},
            {Vectors.dense(4, 1.5), Vectors.dense(4, 0)},
            {Vectors.dense(4, 4)}
        ]

        self.euclidean_ward_count_five_window_as_two_result = [
            {Vectors.dense(1, 1), Vectors.dense(1, 0)},
            {Vectors.dense(1, 4), Vectors.dense(4, 4), Vectors.dense(4, 1.5)}
        ]

        self.euclidean_ward_event_time_window_as_two_result = [
            {Vectors.dense(1, 1), Vectors.dense(1, 0)},
            {Vectors.dense(1, 4)},
            {Vectors.dense(4, 1.5), Vectors.dense(4, 0)},
            {Vectors.dense(4, 4)}
        ]

        self.tolerance = 1e-7

    def test_param(self):
        agglomerative_clustering = AgglomerativeClustering()
        self.assertEqual('features', agglomerative_clustering.features_col)
        self.assertEqual(2, agglomerative_clustering.num_clusters)
        self.assertIsNone(agglomerative_clustering.distance_threshold)
        self.assertEqual('ward', agglomerative_clustering.linkage)
        self.assertEqual('euclidean', agglomerative_clustering.distance_measure)
        self.assertFalse(agglomerative_clustering.compute_full_tree)
        self.assertEqual('prediction', agglomerative_clustering.prediction_col)
        self.assertEqual(GlobalWindows(), agglomerative_clustering.windows)

        agglomerative_clustering \
            .set_features_col("test_features") \
            .set_num_clusters(None) \
            .set_distance_threshold(0.01) \
            .set_linkage('average') \
            .set_distance_measure('cosine') \
            .set_compute_full_tree(True) \
            .set_prediction_col('cluster_id') \
            .set_windows(EventTimeTumblingWindows.of(Time.milliseconds(100)))

        self.assertEqual('test_features', agglomerative_clustering.features_col)
        self.assertIsNone(agglomerative_clustering.num_clusters)
        self.assertEqual(0.01, agglomerative_clustering.distance_threshold)
        self.assertEqual('average', agglomerative_clustering.linkage)
        self.assertEqual('cosine', agglomerative_clustering.distance_measure)
        self.assertTrue(agglomerative_clustering.compute_full_tree)
        self.assertEqual('cluster_id', agglomerative_clustering.prediction_col)
        self.assertEqual(EventTimeTumblingWindows.of(Time.milliseconds(100)),
                         agglomerative_clustering.get_windows())

    def test_output_schema(self):
        input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ('', ''),
            ],
                type_info=Types.ROW_NAMED(
                    ['test_input', 'dummy_input'],
                    [Types.STRING(), Types.STRING()])))

        agglomerative_clustering = AgglomerativeClustering() \
            .set_features_col("test_input") \
            .set_prediction_col("test_prediction")

        outputs = agglomerative_clustering \
            .transform(input_data_table)

        self.assertEqual(2, len(outputs))

        self.assertEqual(
            ['test_input', 'dummy_input', 'test_prediction'],
            outputs[0].get_schema().get_field_names())

        self.assertEqual(
            ['clusterId1', 'clusterId2', 'distance', 'sizeOfMergedCluster'],
            outputs[1].get_schema().get_field_names())

    def verify_clustering_result(self, expected, output_table, features_col, prediction_col):
        predicted_results = [result for result in
                             self.t_env.to_data_stream(output_table).execute_and_collect()]
        field_names = output_table.get_schema().get_field_names()
        actual_groups = group_features_by_prediction(
            predicted_results,
            field_names.index(features_col),
            field_names.index(prediction_col))

        self.assertTrue(expected == actual_groups)

    def verify_merge_info(self, expected, output_table):
        merge_infos = [result for result in
                       self.t_env.to_data_stream(output_table).execute_and_collect()]

        self.assertEquals(len(expected), len(merge_infos))
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], merge_infos[i][2], delta=self.tolerance)

    def test_transform(self):
        agglomerative_clustering = AgglomerativeClustering() \
            .set_linkage('ward') \
            .set_distance_measure('euclidean') \
            .set_prediction_col('pred')

        # Tests euclidean distance with linkage as ward, num_clusters = 2.
        outputs = agglomerative_clustering.transform(self.input_table)
        self.verify_clustering_result(self.euclidean_ward_num_clusters_as_two_result,
                                      outputs[0], "features", "pred")

        # Tests euclidean distance with linkage as ward, num_clusters = 2,
        # compute_full_tree = true.
        outputs = agglomerative_clustering.set_compute_full_tree(True).transform(self.input_table)
        self.verify_clustering_result(self.euclidean_ward_num_clusters_as_two_result,
                                      outputs[0], "features", "pred")

        # Tests euclidean distance with linkage as ward, distance_threshold = 2.
        outputs = agglomerative_clustering \
            .set_num_clusters(None) \
            .set_distance_threshold(2.0) \
            .transform(self.input_table)
        self.verify_clustering_result(self.euclidean_ward_threshold_as_two_result,
                                      outputs[0], "features", "pred")

    def test_transform_with_count_tumbling_windows(self):
        self.env.set_parallelism(1)

        input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([1, 1]),),
                (Vectors.dense([1, 4]),),
                (Vectors.dense([1, 0]),),
                (Vectors.dense([4, 1.5]),),
                (Vectors.dense([4, 4]),),
                (Vectors.dense([4, 0]),),
            ],
                type_info=Types.ROW_NAMED(
                    ['features'],
                    [DenseVectorTypeInfo()])))

        agglomerative_clustering = AgglomerativeClustering() \
            .set_linkage('ward') \
            .set_distance_measure('euclidean') \
            .set_prediction_col('pred') \
            .set_windows(CountTumblingWindows.of(5))

        outputs = agglomerative_clustering.transform(input_table)
        self.verify_clustering_result(self.euclidean_ward_count_five_window_as_two_result,
                                      outputs[0], "features", "pred")

    def test_transform_with_event_time_tumbling_windows(self):
        self.env.set_parallelism(1)

        dense_vector_serializer = get_gateway().jvm.org.apache.flink.table.types.logical.RawType(
            get_gateway().jvm.org.apache.flink.ml.linalg.DenseVector(0).getClass(),
            get_gateway().jvm.org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer()
        ).getSerializerString()

        schema = Schema.new_builder() \
            .column("features", "RAW('org.apache.flink.ml.linalg.DenseVector', '{serializer}')"
                    .format(serializer=dense_vector_serializer)) \
            .column("ts", "TIMESTAMP_LTZ(3)") \
            .watermark("ts", "ts - INTERVAL '3' SECOND") \
            .build()

        input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([1, 1]), Instant.of_epoch_milli(1000),),
                (Vectors.dense([1, 4]), Instant.of_epoch_milli(1000),),
                (Vectors.dense([1, 0]), Instant.of_epoch_milli(1000),),
                (Vectors.dense([4, 1.5]), Instant.of_epoch_milli(4000),),
                (Vectors.dense([4, 4]), Instant.of_epoch_milli(4000),),
                (Vectors.dense([4, 0]), Instant.of_epoch_milli(4000),),
            ],
                type_info=Types.ROW_NAMED(
                    ['features', 'ts'],
                    [DenseVectorTypeInfo(), Types.INSTANT()])),
            schema)

        agglomerative_clustering = AgglomerativeClustering() \
            .set_linkage('ward') \
            .set_distance_measure('euclidean') \
            .set_prediction_col('pred') \
            .set_windows(EventTimeTumblingWindows.of(Time.seconds(1)))

        outputs = agglomerative_clustering.transform(input_table)

        # TODO: remove this line when Instant values can be acquired without errors.
        outputs[0] = outputs[0].select(col("features"), col("pred"))

        predicted_results = [result for result in
                             self.t_env.to_data_stream(outputs[0]).execute_and_collect()]
        field_names = outputs[0].get_schema().get_field_names()
        actual_groups = group_features_by_prediction(
            predicted_results,
            field_names.index("features"),
            field_names.index("pred"))

        is_all_subset = True
        for expected_set in self.euclidean_ward_event_time_window_as_two_result:
            is_subset = False
            for actual_set in actual_groups:
                if expected_set.issubset(actual_set):
                    is_subset = True
                    break
            is_all_subset &= is_subset

        self.assertTrue(is_all_subset)

    def test_merge_info(self):
        agglomerative_clustering = AgglomerativeClustering() \
            .set_linkage('average') \
            .set_distance_measure('euclidean') \
            .set_prediction_col('pred') \
            .set_compute_full_tree(True)

        # Tests euclidean distance with linkage as average.
        outputs = agglomerative_clustering.transform(self.input_table)
        self.verify_merge_info(self.euclidean_average_merge_distances, outputs[1])

        # Tests cosine distance with linkage as average.
        outputs = agglomerative_clustering \
            .set_distance_measure('cosine') \
            .transform(self.input_table)
        self.verify_merge_info(self.cosine_average_merge_distances, outputs[1])

        # Tests manhattan distance with linkage as average.
        outputs = agglomerative_clustering \
            .set_distance_measure('manhattan') \
            .transform(self.input_table)
        self.verify_merge_info(self.manhattan_average_merge_distances, outputs[1])

        # Tests euclidean distance with linkage as complete.
        outputs = agglomerative_clustering \
            .set_distance_measure('euclidean') \
            .set_linkage('complete') \
            .transform(self.input_table)
        self.verify_merge_info(self.euclidean_complete_merge_distances, outputs[1])

        # Tests euclidean distance with linkage as single.
        outputs = agglomerative_clustering.set_linkage('single').transform(self.input_table)
        self.verify_merge_info(self.euclidean_single_merge_distances, outputs[1])

        # Tests euclidean distance with linkage as ward.
        outputs = agglomerative_clustering.set_linkage('ward').transform(self.input_table)
        self.verify_merge_info(self.euclidean_ward_merge_distances, outputs[1])

        # Tests merge info not fully computed.
        outputs = agglomerative_clustering.set_compute_full_tree(False).transform(self.input_table)
        self.verify_merge_info(self.euclidean_ward_merge_distances[0:4], outputs[1])

    def test_save_load_transform(self):
        agglomerative_clustering = AgglomerativeClustering() \
            .set_linkage('ward') \
            .set_distance_measure('euclidean') \
            .set_prediction_col('pred')

        path = os.path.join(self.temp_dir, 'test_save_load_and_transform_agglomerativeclustering')
        agglomerative_clustering.save(path)
        loaded_agglomerative_clustering = AgglomerativeClustering.load(self.t_env, path)
        outputs = loaded_agglomerative_clustering.transform(self.input_table)
        self.verify_clustering_result(self.euclidean_ward_num_clusters_as_two_result,
                                      outputs[0], "features", "pred")
