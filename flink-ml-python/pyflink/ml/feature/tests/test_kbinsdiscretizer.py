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

import numpy as np
from pyflink.common import Types

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.kbinsdiscretizer import KBinsDiscretizer, KBinsDiscretizerModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params


class KBinsDiscretizerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(KBinsDiscretizerTest, self).setUp()
        self.train_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(1, 10, 0),),
                (Vectors.dense(1, 10, 0),),
                (Vectors.dense(1, 10, 0),),
                (Vectors.dense(4, 10, 0),),
                (Vectors.dense(5, 10, 0),),
                (Vectors.dense(6, 10, 0),),
                (Vectors.dense(7, 10, 0),),
                (Vectors.dense(10, 10, 0),),
                (Vectors.dense(13, 10, 3),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input', ],
                    [DenseVectorTypeInfo(), ])))

        self.predict_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(-1, 0, 0),),
                (Vectors.dense(1, 1, 1),),
                (Vectors.dense(1.5, 1, 2),),
                (Vectors.dense(5, 2, 3),),
                (Vectors.dense(7.25, 3, 4),),
                (Vectors.dense(13, 4, 5),),
                (Vectors.dense(15, 4, 6),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input', ],
                    [DenseVectorTypeInfo(), ])))

        self.uniform_output = [
            Vectors.dense(0, 0, 0),
            Vectors.dense(0, 0, 1),
            Vectors.dense(0, 0, 2),
            Vectors.dense(1, 0, 2),
            Vectors.dense(1, 0, 2),
            Vectors.dense(2, 0, 2),
            Vectors.dense(2, 0, 2),
        ]

        self.quantile_output = [
            Vectors.dense(0, 0, 0),
            Vectors.dense(0, 0, 0),
            Vectors.dense(0, 0, 1),
            Vectors.dense(1, 0, 1),
            Vectors.dense(2, 0, 1),
            Vectors.dense(2, 0, 1),
            Vectors.dense(2, 0, 1),
        ]

        self.kmeans_output = [
            Vectors.dense(0, 0, 0),
            Vectors.dense(0, 0, 1),
            Vectors.dense(0, 0, 2),
            Vectors.dense(1, 0, 2),
            Vectors.dense(1, 0, 2),
            Vectors.dense(2, 0, 2),
            Vectors.dense(2, 0, 2),
        ]

        self.eps = 1e-7

    def test_param(self):
        k_bins_discretizer = KBinsDiscretizer()

        self.assertEqual("input", k_bins_discretizer.input_col)
        self.assertEqual(5, k_bins_discretizer.num_bins)
        self.assertEqual("quantile", k_bins_discretizer.strategy)
        self.assertEqual(200000, k_bins_discretizer.sub_samples)
        self.assertEqual("output", k_bins_discretizer.output_col)

        k_bins_discretizer \
            .set_input_col("test_input") \
            .set_num_bins(10) \
            .set_strategy('kmeans') \
            .set_sub_samples(1000) \
            .set_output_col("test_output")

        self.assertEqual("test_input", k_bins_discretizer.input_col)
        self.assertEqual(10, k_bins_discretizer.num_bins)
        self.assertEqual("kmeans", k_bins_discretizer.strategy)
        self.assertEqual(1000, k_bins_discretizer.sub_samples)
        self.assertEqual("test_output", k_bins_discretizer.output_col)

    def test_output_schema(self):
        k_bins_discretizer = KBinsDiscretizer() \
            .set_input_col("test_input") \
            .set_output_col("test_output")
        input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ('', ''),
            ],
                type_info=Types.ROW_NAMED(
                    ['test_input', 'dummy_input'],
                    [Types.STRING(), Types.STRING()])))
        output = k_bins_discretizer \
            .fit(input_data_table) \
            .transform(input_data_table)[0]

        self.assertEqual(
            [k_bins_discretizer.input_col, 'dummy_input', k_bins_discretizer.output_col],
            output.get_schema().get_field_names())

    def verify_prediction_result(self, expected, output_table):
        predicted_results = [result[1] for result in
                             self.t_env.to_data_stream(output_table).execute_and_collect()]

        predicted_results.sort(key=lambda x: (x[0], x[1], x[2]))
        expected.sort(key=lambda x: (x[0], x[1], x[2]))

        self.assertEqual(expected, predicted_results)

    def test_fit_and_predict(self):
        k_bins_discretizer = KBinsDiscretizer().set_num_bins(3)

        # Tests uniform strategy.
        k_bins_discretizer.set_strategy('uniform')
        output = k_bins_discretizer.fit(self.train_table).transform(self.predict_table)[0]
        self.verify_prediction_result(self.uniform_output, output)

        # Tests quantile strategy.
        k_bins_discretizer.set_strategy('quantile')
        output = k_bins_discretizer.fit(self.train_table).transform(self.predict_table)[0]
        self.verify_prediction_result(self.quantile_output, output)

        # Tests kmeans strategy.
        k_bins_discretizer.set_strategy('kmeans')
        output = k_bins_discretizer.fit(self.train_table).transform(self.predict_table)[0]
        self.verify_prediction_result(self.kmeans_output, output)

    def test_get_model_data(self):
        k_bins_discretizer = KBinsDiscretizer().set_num_bins(3).set_strategy('uniform')
        model = k_bins_discretizer.fit(self.train_table)
        model_data = model.get_model_data()[0]
        expected_field_names = ['binEdges']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        bin_edges = model_rows[0][expected_field_names.index('binEdges')]
        self.assertEqual(3, len(bin_edges))
        self.assertListEqual([1, 5, 9, 13], bin_edges[0])
        self.assertListEqual([-np.inf, np.inf], bin_edges[1])
        self.assertListEqual([0, 1, 2, 3], bin_edges[2])

    def test_set_model_data(self):
        k_bins_discretizer = KBinsDiscretizer().set_num_bins(3).set_strategy('uniform')
        model_a = k_bins_discretizer.fit(self.train_table)
        model_data = model_a.get_model_data()[0]

        model_b = KBinsDiscretizerModel().set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output = model_b.transform(self.predict_table)[0]
        self.verify_prediction_result(self.uniform_output, output)

    def test_save_load_predict(self):
        k_bins_discretizer = KBinsDiscretizer().set_num_bins(3)
        estimator_path = os.path.join(self.temp_dir, 'test_save_load_predict_kbinsdiscretizer')
        k_bins_discretizer.save(estimator_path)
        k_bins_discretizer = KBinsDiscretizer.load(self.t_env, estimator_path)

        model = k_bins_discretizer.fit(self.train_table)
        model_path = os.path.join(self.temp_dir, 'test_save_load_predict_kbinsdiscretizer_model')
        model.save(model_path)
        self.env.execute('save_model')
        model = KBinsDiscretizerModel.load(self.t_env, model_path)

        output = model.transform(self.predict_table)[0]
        self.verify_prediction_result(self.quantile_output, output)
