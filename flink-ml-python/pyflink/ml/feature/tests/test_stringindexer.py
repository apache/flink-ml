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

from pyflink.ml.feature.stringindexer import StringIndexer, StringIndexerModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params


class StringIndexerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(StringIndexerTest, self).setUp()
        self.train_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ('a', 1.0),
                ('b', 1.0),
                ('b', 2.0),
                ('c', 0.0),
                ('d', 2.0),
                ('a', 2.0),
                ('b', 2.0),
                ('b', -1.0),
                ('a', -1.0),
                ('c', -1.0),
                ('d', None),
                (None, 2.0),
                (None, None),
            ],
                type_info=Types.ROW_NAMED(
                    ['input_col1', 'input_col2'],
                    [Types.STRING(), Types.DOUBLE()])))

        self.predict_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ('a', 2.0),
                ('b', 1.0),
                ('e', 2.0),
                ('f', None),
                (None, None),
            ],
                type_info=Types.ROW_NAMED(
                    ['input_col1', 'input_col2'],
                    [Types.STRING(), Types.DOUBLE()])))

        self.expected_alphabetic_asc_predict_data = [
            Row('a', 2.0, 0, 3),
            Row('b', 1.0, 1, 2),
            Row('e', 2.0, 4, 3),
            Row('f', None, 4, 4),
            Row(None, None, 4, 4),
        ]

    def test_param(self):
        string_indexer = StringIndexer()

        self.assertEqual('arbitrary', string_indexer.string_order_type)
        self.assertEqual('error', string_indexer.handle_invalid)

        string_indexer.set_input_cols('input_col1', 'input_col2') \
            .set_output_cols('output_col1', 'output_col2') \
            .set_string_order_type('alphabetAsc') \
            .set_handle_invalid('skip')

        self.assertEqual(('input_col1', 'input_col2'), string_indexer.input_cols)
        self.assertEqual(('output_col1', 'output_col2'), string_indexer.output_cols)
        self.assertEqual('alphabetAsc', string_indexer.string_order_type)
        self.assertEqual('skip', string_indexer.handle_invalid)

    def test_output_schema(self):
        string_indexer = StringIndexer() \
            .set_input_cols('input_col1', 'input_col2') \
            .set_output_cols('output_col1', 'output_col2') \
            .set_string_order_type('alphabetAsc') \
            .set_handle_invalid('skip')

        output = string_indexer.fit(self.train_table).transform(self.predict_table)[0]

        self.assertEqual(
            ['input_col1', 'input_col2', 'output_col1', 'output_col2'],
            output.get_schema().get_field_names())

    def test_string_order_type(self):
        string_indexer = StringIndexer() \
            .set_input_cols('input_col1', 'input_col2') \
            .set_output_cols('output_col1', 'output_col2') \
            .set_handle_invalid('keep')

        string_indexer.set_string_order_type('alphabetAsc')
        output = string_indexer.fit(self.train_table).transform(self.predict_table)[0]

        predicted_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]

        predicted_results.sort(key=lambda x: (x[0] is None, x[0]))

        self.assertEqual(predicted_results, self.expected_alphabetic_asc_predict_data)

    def test_fit_and_predict(self):
        string_indexer = StringIndexer() \
            .set_input_cols('input_col1', 'input_col2') \
            .set_output_cols('output_col1', 'output_col2') \
            .set_string_order_type('alphabetAsc') \
            .set_handle_invalid('keep')

        output = string_indexer.fit(self.train_table).transform(self.predict_table)[0]

        predicted_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]

        predicted_results.sort(key=lambda x: (x[0] is None, x[0]))

        self.assertEqual(predicted_results, self.expected_alphabetic_asc_predict_data)

    def test_get_model_data(self):
        string_indexer = StringIndexer() \
            .set_input_cols('input_col1', 'input_col2') \
            .set_output_cols('output_col1', 'output_col2') \
            .set_string_order_type('alphabetAsc')
        model = string_indexer.fit(self.train_table)
        model_data = model.get_model_data()[0]
        expected_field_names = ['stringArrays']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        string_arrays = model_rows[0][expected_field_names.index('stringArrays')]
        self.assertListEqual(["a", "b", "c", "d"], string_arrays[0])
        self.assertListEqual(["-1.0", "0.0", "1.0", "2.0"], string_arrays[1])

    def test_set_model_data(self):
        string_indexer = StringIndexer() \
            .set_input_cols('input_col1', 'input_col2') \
            .set_output_cols('output_col1', 'output_col2') \
            .set_string_order_type('alphabetAsc') \
            .set_handle_invalid('keep')
        model_a = string_indexer.fit(self.train_table)
        model_data = model_a.get_model_data()[0]

        model_b = StringIndexerModel().set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output = model_b.transform(self.predict_table)[0]

        predicted_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]

        predicted_results.sort(key=lambda x: (x[0] is None, x[0]))

        self.assertEqual(predicted_results, self.expected_alphabetic_asc_predict_data)

    def test_save_load_and_predict(self):
        string_indexer = StringIndexer() \
            .set_input_cols('input_col1', 'input_col2') \
            .set_output_cols('output_col1', 'output_col2') \
            .set_string_order_type('alphabetAsc') \
            .set_handle_invalid('keep')
        reloaded_string_indexer = self.save_and_reload(string_indexer)

        model = reloaded_string_indexer.fit(self.train_table)
        reloaded_model = self.save_and_reload(model)

        output = reloaded_model.transform(self.predict_table)[0]
        predicted_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]
        predicted_results.sort(key=lambda x: (x[0] is None, x[0]))
        self.assertEqual(predicted_results, self.expected_alphabetic_asc_predict_data)
