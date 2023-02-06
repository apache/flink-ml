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
from pyflink.ml.feature.variancethresholdselector import \
    VarianceThresholdSelector, VarianceThresholdSelectorModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params


class VarianceThresholdSelectorTest(PyFlinkMLTestCase):
    def setUp(self):
        super(VarianceThresholdSelectorTest, self).setUp()
        self.train_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (1, Vectors.dense(5.0, 7.0, 0.0, 7.0, 6.0, 0.0),),
                (2, Vectors.dense(0.0, 9.0, 6.0, 0.0, 5.0, 9.0),),
                (3, Vectors.dense(0.0, 9.0, 3.0, 0.0, 5.0, 5.0),),
                (4, Vectors.dense(1.0, 9.0, 8.0, 5.0, 7.0, 4.0),),
                (5, Vectors.dense(9.0, 8.0, 6.0, 5.0, 4.0, 4.0),),
                (6, Vectors.dense(6.0, 9.0, 7.0, 0.0, 2.0, 0.0),),
            ],
                type_info=Types.ROW_NAMED(
                    ['id', 'input'],
                    [Types.INT(), DenseVectorTypeInfo()])
            ))

        self.predict_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),),
                (Vectors.dense(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [DenseVectorTypeInfo()])
            ))
        self.expected_output = [
            Vectors.dense(0.1, 0.4, 0.6),
            Vectors.dense(1.0, 4.0, 6.0)]

    def test_param(self):
        variance_threshold_selector = VarianceThresholdSelector()
        self.assertEqual("input", variance_threshold_selector.input_col)
        self.assertEqual("output", variance_threshold_selector.output_col)
        self.assertEqual(0.0, variance_threshold_selector.variance_threshold)

        variance_threshold_selector.\
            set_input_col("test_input").\
            set_output_col("test_output").\
            set_variance_threshold(8.0)
        self.assertEqual("test_input", variance_threshold_selector.input_col)
        self.assertEqual("test_output", variance_threshold_selector.output_col)
        self.assertEqual(8.0, variance_threshold_selector.variance_threshold)

    def test_output_schema(self):
        variance_threshold_selector = VarianceThresholdSelector() \
            .set_output_col('test_output') \
            .set_variance_threshold(8.0)

        model = variance_threshold_selector.fit(self.train_table)
        output = model.transform(self.predict_table.alias('test_input'))[0]
        self.assertEqual(
            ['test_input', 'test_output'],
            output.get_schema().get_field_names())

    def test_fit_and_predict(self):
        variance_threshold_selector = VarianceThresholdSelector() \
            .set_variance_threshold(8.0)
        model = variance_threshold_selector.fit(self.train_table)
        output = model.transform(self.predict_table)[0]
        self.verify_output_result(
            output,
            variance_threshold_selector.get_output_col(),
            output.get_schema().get_field_names(),
            self.expected_output)

    def test_incompatible_num_of_features(self):
        variance_threshold_selector = VarianceThresholdSelector() \
            .set_variance_threshold(8.0)
        model = variance_threshold_selector.fit(self.train_table)
        predict_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0),),
                (Vectors.dense(0.1, 0.2, 0.3, 0.4, 0.5),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [DenseVectorTypeInfo()])
            ))
        with self.assertRaisesRegex(Exception, 'but VarianceThresholdSelector is expecting'):
            output = model.transform(predict_table)[0]
            self.verify_output_result(
                output,
                variance_threshold_selector.get_output_col(),
                output.get_schema().get_field_names(),
                self.expected_output)

    def test_get_model_data(self):
        variance_threshold_selector = VarianceThresholdSelector() \
            .set_variance_threshold(8.0)
        model = variance_threshold_selector.fit(self.train_table)
        model_data = model.get_model_data()[0]
        expected_field_names = ['numOfFeatures', 'indices']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        self.assertEqual(6, model_rows[0][expected_field_names.index('numOfFeatures')])
        self.assertListEqual([0, 3, 5], model_rows[0][expected_field_names.index('indices')])

    def test_set_model_data(self):
        variance_threshold_selector = VarianceThresholdSelector() \
            .set_variance_threshold(8.0)
        model_a = variance_threshold_selector.fit(self.train_table)
        model_data = model_a.get_model_data()[0]

        model_b = VarianceThresholdSelectorModel().set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output = model_b.transform(self.predict_table)[0]
        self.verify_output_result(
            output,
            variance_threshold_selector.get_output_col(),
            output.get_schema().get_field_names(),
            self.expected_output)

    def test_save_load_predict(self):
        variance_threshold_selector = VarianceThresholdSelector() \
            .set_variance_threshold(8.0)
        reloaded_variance_threshold_selector = self.save_and_reload(variance_threshold_selector)
        model = reloaded_variance_threshold_selector.fit(self.train_table)
        reloaded_model = self.save_and_reload(model)
        output = reloaded_model.transform(self.predict_table)[0]
        self.verify_output_result(
            output,
            variance_threshold_selector.get_output_col(),
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
