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
from typing import List

from pyflink.common import Types
from pyflink.table import Table

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo, DenseVector
from pyflink.ml.feature.minmaxscaler import MinMaxScaler, MinMaxScalerModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params


class MinMaxScalerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(MinMaxScalerTest, self).setUp()
        self.train_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([0.0, 3.0]),),
                (Vectors.dense([2.1, 0.0]),),
                (Vectors.dense([4.1, 5.1]),),
                (Vectors.dense([6.1, 8.1]),),
                (Vectors.dense([200., 400.]),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [DenseVectorTypeInfo()])))

        self.predict_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([150.0, 90.0]),),
                (Vectors.dense([50.0, 40.0]),),
                (Vectors.dense([100.0, 50.0]),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [DenseVectorTypeInfo()])))
        self.expected_data = [
            Vectors.dense(0.25, 0.1),
            Vectors.dense(0.5, 0.125),
            Vectors.dense(0.75, 0.225)]

    def test_param(self):
        min_max_scaler = MinMaxScaler()
        self.assertEqual("input", min_max_scaler.input_col)
        self.assertEqual("output", min_max_scaler.output_col)
        self.assertEqual(0.0, min_max_scaler.min)
        self.assertEqual(1.0, min_max_scaler.max)
        min_max_scaler.set_input_col('test_input') \
            .set_output_col('test_output') \
            .set_min(1.0) \
            .set_max(4.0)
        self.assertEqual('test_input', min_max_scaler.input_col)
        self.assertEqual(1.0, min_max_scaler.min)
        self.assertEqual(4.0, min_max_scaler.max)
        self.assertEqual('test_output', min_max_scaler.output_col)

    def test_output_schema(self):
        min_max_scaler = MinMaxScaler() \
            .set_input_col('test_input') \
            .set_output_col('test_output') \
            .set_min(1.0) \
            .set_max(4.0)

        model = min_max_scaler.fit(self.train_data.alias('test_input'))
        output = model.transform(self.predict_data.alias('test_input'))[0]
        self.assertEqual(
            ['test_input', 'test_output'],
            output.get_schema().get_field_names())

    def test_max_value_equas_min_value_but_predict_value_not_equals(self):
        train_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([40.0, 80.0]),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [DenseVectorTypeInfo()])))

        predict_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([30.0, 50.0]),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [DenseVectorTypeInfo()])))

        min_max_scaler = MinMaxScaler() \
            .set_min(0.0) \
            .set_max(10.0)

        model = min_max_scaler.fit(train_data)
        result = model.transform(predict_data)[0]
        self.verify_output_result(
            result,
            min_max_scaler.get_output_col(),
            result.get_schema().get_field_names(),
            [Vectors.dense(5.0, 5.0)])

    def test_fit_and_predict(self):
        min_max_scaler = MinMaxScaler()
        model = min_max_scaler.fit(self.train_data)
        output = model.transform(self.predict_data)[0]
        self.verify_output_result(
            output,
            min_max_scaler.get_output_col(),
            output.get_schema().get_field_names(),
            self.expected_data)

    def test_get_model_data(self):
        min_max_scaler = MinMaxScaler()
        model = min_max_scaler.fit(self.train_data)
        model_data = model.get_model_data()[0]
        expected_field_names = ['minVector', 'maxVector']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        self.assertListAlmostEqual(
            [0.0, 0.0], model_rows[0][expected_field_names.index('minVector')])
        self.assertListAlmostEqual(
            [200.0, 400.0], model_rows[0][expected_field_names.index('maxVector')])

    def test_set_model_data(self):
        min_max_scaler = MinMaxScaler()
        model_a = min_max_scaler.fit(self.train_data)
        model_data = model_a.get_model_data()[0]

        model_b = MinMaxScalerModel().set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output = model_b.transform(self.predict_data)[0]
        self.verify_output_result(
            output,
            min_max_scaler.get_output_col(),
            output.get_schema().get_field_names(),
            self.expected_data)

    def test_save_load_and_predict(self):
        min_max_scaler = MinMaxScaler()
        reloaded_min_max_scaler = self.save_and_reload(min_max_scaler)
        model = reloaded_min_max_scaler.fit(self.train_data)
        reloaded_model = self.save_and_reload(model)
        output = reloaded_model.transform(self.predict_data)[0]
        self.verify_output_result(
            output,
            min_max_scaler.get_output_col(),
            output.get_schema().get_field_names(),
            self.expected_data)

    def verify_output_result(
            self, output: Table,
            output_col: str,
            field_names: List[str],
            expected_result: List[DenseVector]):
        collected_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]
        results = []
        for item in collected_results:
            item.set_field_names(field_names)
            results.append(item[output_col])
        results.sort(key=lambda x: x[0])
        self.assertEqual(expected_result, results)
