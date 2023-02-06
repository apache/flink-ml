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
import typing
from pyflink.common import Types, Row
from typing import List, Dict, Set

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo, DenseVector
from pyflink.ml.clustering.kmeans import KMeans, KMeansModel, OnlineKMeans
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params


def group_features_by_prediction(
        rows: List[Row], feature_index: int, prediction_index: int):
    map = {}  # type: Dict[int, Set]
    for row in rows:
        vector = typing.cast(DenseVector, row[feature_index])
        predict = typing.cast(int, row[prediction_index])
        if predict in map:
            l = map[predict]
        else:
            l = set()
            map[predict] = l
        l.add(vector)
    return [item for item in map.values()]


class KMeansTest(PyFlinkMLTestCase):
    def setUp(self):
        super(KMeansTest, self).setUp()
        self.data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([0.0, 0.0]),),
                (Vectors.dense([0.0, 0.3]),),
                (Vectors.dense([0.3, 0.0]),),
                (Vectors.dense([9.0, 0.0]),),
                (Vectors.dense([9.0, 0.6]),),
                (Vectors.dense([9.6, 0.0]),),
            ],
                type_info=Types.ROW_NAMED(
                    ['features'],
                    [DenseVectorTypeInfo()])))
        self.expected_groups = [
            {DenseVector([0.0, 0.3]), DenseVector([0.3, 0.0]), DenseVector([0.0, 0.0])},
            {DenseVector([9.6, 0.0]), DenseVector([9.0, 0.0]), DenseVector([9.0, 0.6])}]

    def test_param(self):
        kmeans = KMeans()
        self.assertEqual('features', kmeans.get_features_col())
        self.assertEqual('prediction', kmeans.get_prediction_col())
        self.assertEqual('euclidean', kmeans.get_distance_measure())
        self.assertEqual('random', kmeans.get_init_mode())
        self.assertEqual(2, kmeans.get_k())
        self.assertEqual(20, kmeans.get_max_iter())

        kmeans.set_k(9) \
            .set_features_col('test_feature') \
            .set_prediction_col('test_prediction') \
            .set_k(3) \
            .set_max_iter(30) \
            .set_seed(100)

        self.assertEqual('test_feature', kmeans.get_features_col())
        self.assertEqual('test_prediction', kmeans.get_prediction_col())
        self.assertEqual(3, kmeans.get_k())
        self.assertEqual(30, kmeans.get_max_iter())
        self.assertEqual(100, kmeans.get_seed())

    def test_output_schema(self):
        input = self.data_table.alias('test_feature')
        kmeans = KMeans().set_features_col('test_feature').set_prediction_col('test_prediction')

        model = kmeans.fit(input)
        output = model.transform(input)[0]

        field_names = output.get_schema().get_field_names()
        self.assertEqual(['test_feature', 'test_prediction'],
                         field_names)

        results = [result for result in self.t_env.to_data_stream(output).execute_and_collect()]
        actual_groups = group_features_by_prediction(
            results,
            field_names.index(kmeans.features_col),
            field_names.index(kmeans.prediction_col))

        self.assertTrue(actual_groups[0] == self.expected_groups[0]
                        and actual_groups[1] == self.expected_groups[1] or
                        actual_groups[0] == self.expected_groups[1]
                        and actual_groups[1] == self.expected_groups[0])

    def test_fewer_distinct_points_than_cluster(self):
        input = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([0.0, 0.1]),),
                (Vectors.dense([0.0, 0.1]),),
                (Vectors.dense([0.0, 0.1]),),
            ],
                type_info=Types.ROW_NAMED(
                    ['features'],
                    [DenseVectorTypeInfo()])))

        kmeans = KMeans().set_k(2)
        model = kmeans.fit(input)
        output = model.transform(input)[0]
        results = [result for result in self.t_env.to_data_stream(output).execute_and_collect()]
        field_names = output.get_schema().get_field_names()
        actual_groups = group_features_by_prediction(
            results,
            field_names.index(kmeans.features_col),
            field_names.index(kmeans.prediction_col))

        expected_groups = [{DenseVector([0.0, 0.1])}]

        self.assertEqual(actual_groups, expected_groups)

    def test_fit_and_predict(self):
        kmeans = KMeans().set_max_iter(2).set_k(2)
        model = kmeans.fit(self.data_table)
        output = model.transform(self.data_table)[0]

        self.assertEqual(['features', 'prediction'], output.get_schema().get_field_names())
        results = [result for result in self.t_env.to_data_stream(output).execute_and_collect()]
        field_names = output.get_schema().get_field_names()
        actual_groups = group_features_by_prediction(
            results,
            field_names.index(kmeans.features_col),
            field_names.index(kmeans.prediction_col))

        self.assertTrue(actual_groups[0] == self.expected_groups[0]
                        and actual_groups[1] == self.expected_groups[1] or
                        actual_groups[0] == self.expected_groups[1]
                        and actual_groups[1] == self.expected_groups[0])

    def test_get_model_data(self):
        kmeans = KMeans().set_max_iter(2).set_k(2)
        model = kmeans.fit(self.data_table)
        model_data = model.get_model_data()[0]
        expected_field_names = ['centroids', 'weights']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        centroids = model_rows[0][expected_field_names.index('centroids')]
        self.assertEqual(2, len(centroids))
        centroids.sort(key=lambda x: x.get(0))
        self.assertListAlmostEqual([0.1, 0.1], centroids[0], delta=1e-5)
        self.assertListAlmostEqual([9.2, 0.2], centroids[1], delta=1e-5)

    def test_set_model_data(self):
        kmeans = KMeans().set_max_iter(2).set_k(2)
        model_a = kmeans.fit(self.data_table)
        model_data = model_a.get_model_data()[0]

        model_b = KMeansModel().set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output = model_b.transform(self.data_table)[0]
        self.assertEqual(['features', 'prediction'], output.get_schema().get_field_names())
        results = [result for result in self.t_env.to_data_stream(output).execute_and_collect()]
        field_names = output.get_schema().get_field_names()
        actual_groups = group_features_by_prediction(
            results,
            field_names.index(kmeans.features_col),
            field_names.index(kmeans.prediction_col))

        self.assertTrue(actual_groups[0] == self.expected_groups[0]
                        and actual_groups[1] == self.expected_groups[1] or
                        actual_groups[0] == self.expected_groups[1]
                        and actual_groups[1] == self.expected_groups[0])

    def test_save_load_and_predict(self):
        kmeans = KMeans().set_max_iter(2).set_k(2)
        model = kmeans.fit(self.data_table)
        path = os.path.join(self.temp_dir, 'test_save_load_and_predict_kmeans_model')
        model.save(path)
        self.env.execute()
        loaded_model = KMeansModel.load(self.t_env, path)  # type: KMeansModel
        output = loaded_model.transform(self.data_table)[0]
        self.assertEqual(
            ['centroids', 'weights'],
            loaded_model.get_model_data()[0].get_schema().get_field_names())

        self.assertEqual(
            ['features', 'prediction'],
            output.get_schema().get_field_names())

        results = [result for result in self.t_env.to_data_stream(output).execute_and_collect()]
        field_names = output.get_schema().get_field_names()
        actual_groups = group_features_by_prediction(
            results,
            field_names.index(kmeans.features_col),
            field_names.index(kmeans.prediction_col))
        self.assertTrue(actual_groups[0] == self.expected_groups[0]
                        and actual_groups[1] == self.expected_groups[1] or
                        actual_groups[0] == self.expected_groups[1]
                        and actual_groups[1] == self.expected_groups[0])


class OnlineKMeansTest(PyFlinkMLTestCase):
    def setUp(self):
        super(OnlineKMeansTest, self).setUp()

    def test_param(self):
        online_kmeans = OnlineKMeans()
        self.assertEqual('features', online_kmeans.features_col)
        self.assertEqual('prediction', online_kmeans.prediction_col)
        self.assertEqual('count', online_kmeans.batch_strategy)
        self.assertEqual('euclidean', online_kmeans.distance_measure)
        self.assertEqual(32, online_kmeans.global_batch_size)
        self.assertEqual(0., online_kmeans.decay_factor)

        online_kmeans.set_features_col('test_feature') \
            .set_prediction_col('test_prediction') \
            .set_global_batch_size(5) \
            .set_decay_factor(0.25) \
            .set_seed(100)

        self.assertEqual('test_feature', online_kmeans.features_col)
        self.assertEqual('test_prediction', online_kmeans.prediction_col)
        self.assertEqual('count', online_kmeans.batch_strategy)
        self.assertEqual('euclidean', online_kmeans.distance_measure)
        self.assertEqual(5, online_kmeans.global_batch_size)
        self.assertEqual(0.25, online_kmeans.decay_factor)
        self.assertEqual(100, online_kmeans.get_seed())
