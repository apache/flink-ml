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
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo, DenseVector

from pyflink.ml.feature.robustscaler import RobustScaler, RobustScalerModel
from pyflink.table import Table


class RobustScalerTest(PyFlinkMLTestCase):

    def setUp(self):
        super(RobustScalerTest, self).setUp()
        self.train_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (1, Vectors.dense(0.0, 0.0),),
                (2, Vectors.dense(1.0, -1.0),),
                (3, Vectors.dense(2.0, -2.0),),
                (4, Vectors.dense(3.0, -3.0),),
                (5, Vectors.dense(4.0, -4.0),),
                (6, Vectors.dense(5.0, -5.0),),
                (7, Vectors.dense(6.0, -6.0),),
                (8, Vectors.dense(7.0, -7.0),),
                (9, Vectors.dense(8.0, -8.0),),
            ],
                type_info=Types.ROW_NAMED(
                    ['id', 'input'],
                    [Types.INT(), DenseVectorTypeInfo()])
            ))

        self.predict_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(3.0, -3.0),),
                (Vectors.dense(6.0, -6.0),),
                (Vectors.dense(99.0, -99.0),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [DenseVectorTypeInfo()])
            ))

        self.expected_output = [
            Vectors.dense(0.75, -0.75),
            Vectors.dense(1.5, -1.5),
            Vectors.dense(24.75, -24.75)]

    def test_param(self):
        robust_scaler = RobustScaler()
        self.assertEqual("input", robust_scaler.input_col)
        self.assertEqual("output", robust_scaler.output_col)
        self.assertEqual(0.25, robust_scaler.lower)
        self.assertEqual(0.75, robust_scaler.upper)
        self.assertEqual(0.001, robust_scaler.relative_error)
        self.assertFalse(robust_scaler.with_centering)
        self.assertTrue(robust_scaler.with_scaling)

        robust_scaler\
            .set_input_col("test_input")\
            .set_output_col("test_output")\
            .set_lower(0.1)\
            .set_upper(0.9)\
            .set_relative_error(0.01)\
            .set_with_centering(True)\
            .set_with_scaling(False)

        self.assertEqual("test_input", robust_scaler.input_col)
        self.assertEqual("test_output", robust_scaler.output_col)
        self.assertEqual(0.1, robust_scaler.lower)
        self.assertEqual(0.9, robust_scaler.upper)
        self.assertEqual(0.01, robust_scaler.relative_error)
        self.assertTrue(robust_scaler.with_centering)
        self.assertFalse(robust_scaler.with_scaling)

    def test_output_schema(self):
        robust_scaler = RobustScaler().set_output_col('test_output')
        model = robust_scaler.fit(self.train_table)
        output = model.transform(self.predict_table.alias('test_input'))[0]
        self.assertEqual(
            ['test_input', 'test_output'],
            output.get_schema().get_field_names())

    def test_fit_and_predict(self):
        robust_scaler = RobustScaler()
        model = robust_scaler.fit(self.train_table)
        output = model.transform(self.predict_table)[0]
        self.verify_output_result(
            output,
            robust_scaler.get_output_col(),
            output.get_schema().get_field_names(),
            self.expected_output)

    def test_get_model_data(self):
        robust_scaler = RobustScaler()
        model = robust_scaler.fit(self.train_table)
        model_data = model.get_model_data()[0]
        expected_field_names = ['medians', 'ranges']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        self.assertListAlmostEqual(
            [4.0, -4.0], model_rows[0][expected_field_names.index('medians')])
        self.assertListAlmostEqual(
            [4.0, 4.0], model_rows[0][expected_field_names.index('ranges')])

    def test_set_model_data(self):
        robust_scaler = RobustScaler()
        model_a = robust_scaler.fit(self.train_table)
        model_data = model_a.get_model_data()[0]

        model_b = RobustScalerModel().set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output = model_b.transform(self.predict_table)[0]
        self.verify_output_result(
            output,
            robust_scaler.get_output_col(),
            output.get_schema().get_field_names(),
            self.expected_output)

    def test_save_load_predict(self):
        robust_scaler = RobustScaler()
        reloaded_robust_scaler = self.save_and_reload(robust_scaler)
        model = reloaded_robust_scaler.fit(self.train_table)
        reloaded_model = self.save_and_reload(model)
        output = reloaded_model.transform(self.predict_table)[0]
        self.verify_output_result(
            output,
            robust_scaler.get_output_col(),
            output.get_schema().get_field_names(),
            self.expected_output)

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
