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

from pyflink.common import Types, Row

from pyflink.ml.feature.stringindexer import IndexToStringModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class IndexToStringModelTest(PyFlinkMLTestCase):
    def setUp(self):
        super(IndexToStringModelTest, self).setUp()
        self.model_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ([['a', 'b', 'c', 'd'], [-1., 0., 1., 2.]],),
            ],
                type_info=Types.ROW_NAMED(
                    ['stringArrays'],
                    [Types.OBJECT_ARRAY(Types.OBJECT_ARRAY(Types.STRING()))])
            ))

        self.predict_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (0, 3),
                (1, 2),
            ],
                type_info=Types.ROW_NAMED(
                    ['input_col1', 'input_col2'],
                    [Types.INT(), Types.INT()])
            ))

        self.expected_prediction = [
            Row(0, 3, 'a', '2.0'),
            Row(1, 2, 'b', '1.0'),
        ]

    def test_output_schema(self):
        model = IndexToStringModel() \
            .set_input_cols('input_col1', 'input_col2') \
            .set_output_cols('output_col1', 'output_col2') \
            .set_model_data(self.model_data_table)

        output = model.transform(self.predict_table)[0]

        self.assertEqual(
            ['input_col1', 'input_col2', 'output_col1', 'output_col2'],
            output.get_schema().get_field_names())

    def test_fit_and_predict(self):
        model = IndexToStringModel() \
            .set_input_cols('input_col1', 'input_col2') \
            .set_output_cols('output_col1', 'output_col2') \
            .set_model_data(self.model_data_table)

        output = model.transform(self.predict_table)[0]

        predicted_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]

        predicted_results.sort(key=lambda x: x[0])

        self.assertEqual(predicted_results, self.expected_prediction)

    def test_get_model_data(self):
        model = IndexToStringModel() \
            .set_input_cols('input_col1', 'input_col2') \
            .set_output_cols('output_col1', 'output_col2') \
            .set_model_data(self.model_data_table)
        model_data = model.get_model_data()[0]
        expected_field_names = ['stringArrays']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        string_arrays = model_rows[0][expected_field_names.index('stringArrays')]
        self.assertListEqual(["a", "b", "c", "d"], string_arrays[0])
        self.assertListEqual(["-1.0", "0.0", "1.0", "2.0"], string_arrays[1])

    def test_save_load_and_predict(self):
        model = IndexToStringModel() \
            .set_input_cols('input_col1', 'input_col2') \
            .set_output_cols('output_col1', 'output_col2') \
            .set_model_data(self.model_data_table)

        reloaded_model = self.save_and_reload(model)

        output = reloaded_model.transform(self.predict_table)[0]

        predicted_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]

        predicted_results.sort(key=lambda x: x[0])

        self.assertEqual(predicted_results, self.expected_prediction)
