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

from pyflink.common import Types
from pyflink.table import Table
from typing import List

from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo, DenseVector
from pyflink.ml.lib.feature.standardscaler import StandardScaler, StandardScalerModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params


class StandardScalerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(StandardScalerTest, self).setUp()
        self.dense_input = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(-2.5, 9.0, 1.0),),
                (Vectors.dense(1.4, -5.0, 1.0),),
                (Vectors.dense(2.0, -1.0, -2.0),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [DenseVectorTypeInfo()])))

        self.expected_res_with_mean = [
            Vectors.dense(-2.8, 8.0, 1.0),
            Vectors.dense(1.1, -6.0, 1.0),
            Vectors.dense(1.7, -2.0, -2.0)
        ]

        self.expected_res_with_std = [
            Vectors.dense(-1.0231819, 1.2480754, 0.5773502),
            Vectors.dense(0.5729819, -0.6933752, 0.5773503),
            Vectors.dense(0.8185455, -0.1386750, -1.1547005)
        ]

        self.expected_res_with_mean_and_std = [
            Vectors.dense(-1.1459637, 1.1094004, 0.5773503),
            Vectors.dense(0.45020003, -0.8320503, 0.5773503),
            Vectors.dense(0.69576368, -0.2773501, -1.1547005)
        ]

        self.expected_mean = [0.3, 1.0, 0.0]
        self.expected_std = [2.4433583, 7.2111026, 1.7320508]

    def test_param(self):
        standard_scaler = StandardScaler()

        self.assertEqual('input', standard_scaler.input_col)
        self.assertEqual(False, standard_scaler.with_mean)
        self.assertEqual(True, standard_scaler.with_std)
        self.assertEqual('output', standard_scaler.output_col)

        standard_scaler.set_input_col('test_input') \
            .set_with_mean(True) \
            .set_with_std(False) \
            .set_output_col('test_output')

        self.assertEqual('test_input', standard_scaler.input_col)
        self.assertEqual(True, standard_scaler.with_mean)
        self.assertEqual(False, standard_scaler.with_std)
        self.assertEqual('test_output', standard_scaler.output_col)

    def test_output_schema(self):
        temp_table = self.dense_input.alias('test_input')

        standard_scaler = StandardScaler().set_input_col('test_input').set_output_col('test_output')

        output = standard_scaler.fit(temp_table).transform(temp_table)[0]

        self.assertEqual(['test_input', 'test_output'], output.get_schema().get_field_names())

    def test_fit_and_predict_with_std(self):
        standard_scaler = StandardScaler()
        output = standard_scaler.fit(self.dense_input).transform(self.dense_input)[0]
        self.verify_output_result(
            output,
            output.get_schema().get_field_names(),
            standard_scaler.get_output_col(),
            self.expected_res_with_std)

    def test_fit_and_predict_with_mean(self):
        standard_scaler = StandardScaler().set_with_std(False).set_with_mean(True)
        output = standard_scaler.fit(self.dense_input).transform(self.dense_input)[0]
        self.verify_output_result(
            output,
            output.get_schema().get_field_names(),
            standard_scaler.get_output_col(),
            self.expected_res_with_mean)

    def test_fit_and_predict_with_mean_and_std(self):
        standard_scaler = StandardScaler().set_with_mean(True)
        output = standard_scaler.fit(self.dense_input).transform(self.dense_input)[0]
        self.verify_output_result(
            output,
            output.get_schema().get_field_names(),
            standard_scaler.get_output_col(),
            self.expected_res_with_mean_and_std)

    def test_get_model_data(self):
        standard_scaler = StandardScaler()
        model = standard_scaler.fit(self.dense_input)
        model_data = model.get_model_data()[0]
        expected_field_names = ['mean', 'std']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names()[0:2])

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        self.assertListAlmostEqual(
            self.expected_mean, model_rows[0][expected_field_names.index('mean')])
        self.assertListAlmostEqual(
            self.expected_std, model_rows[0][expected_field_names.index('std')])

    def test_set_model_data(self):
        standard_scaler = StandardScaler()
        model_a = standard_scaler.fit(self.dense_input)
        model_data = model_a.get_model_data()[0]

        model_b = StandardScalerModel().set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output = model_b.transform(self.dense_input)[0]
        self.verify_output_result(
            output,
            output.get_schema().get_field_names(),
            standard_scaler.get_output_col(),
            self.expected_res_with_std)

    def test_save_load_and_predict(self):
        standard_scaler = StandardScaler()
        reloaded_standard_scaler = self.save_and_reload(standard_scaler)
        model = reloaded_standard_scaler.fit(self.dense_input)
        reloaded_model = self.save_and_reload(model)
        output = reloaded_model.transform(self.dense_input)[0]
        self.verify_output_result(
            output,
            output.get_schema().get_field_names(),
            standard_scaler.get_output_col(),
            self.expected_res_with_std)

    def verify_output_result(
            self,
            output: Table,
            field_names: List[str],
            prediction_col: str,
            expected_result: List[DenseVector]):
        collected_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]

        results = []  # type: List[DenseVector]
        for item in collected_results:
            item.set_field_names(field_names)
            results.append(item[prediction_col])
        results.sort(key=lambda x: x[0])

        for item1, item2 in zip(results, expected_result):
            for i, j in zip(item1._values, item2._values):
                self.assertAlmostEqual(i, j, delta=1e-7)
