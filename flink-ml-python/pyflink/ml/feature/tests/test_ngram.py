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

from pyflink.ml.feature.ngram import NGram
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class NGramTest(PyFlinkMLTestCase):
    def setUp(self):
        super(NGramTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ([],),
                (['a', 'b', 'c'],),
                (['a', 'b', 'c', 'd'],),
            ],
                type_info=Types.ROW_NAMED(
                    ["input", ],
                    [Types.OBJECT_ARRAY(Types.STRING())])))

        self.expected_output = [
            [],
            ['a b', 'b c'],
            ['a b', 'b c', 'c d']
        ]

    def test_param(self):
        n_gram = NGram()
        self.assertEqual('input', n_gram.input_col)
        self.assertEqual('output', n_gram.output_col)
        self.assertEqual(2, n_gram.n)

        n_gram.set_input_col("test_input_col") \
            .set_output_col("test_output_col") \
            .set_n(5)

        self.assertEqual('test_input_col', n_gram.input_col)
        self.assertEqual('test_output_col', n_gram.output_col)
        self.assertEqual(5, n_gram.n)

    def test_output_schema(self):
        n_gram = NGram()
        input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ([''], ''),
            ],
                type_info=Types.ROW_NAMED(
                    ['input', 'dummy_input'],
                    [Types.OBJECT_ARRAY(Types.STRING()), Types.STRING()])))

        output = n_gram.transform(input_data_table)[0]

        self.assertEqual([n_gram.input_col, 'dummy_input',
                          n_gram.output_col], output.get_schema().get_field_names())

    def verify_output_result(self, output_table):
        predicted_result = [result[1] for result in
                            self.t_env.to_data_stream(output_table).execute_and_collect()]
        predicted_result.sort(key=lambda x: len(x))
        self.assertEqual(len(self.expected_output), len(predicted_result))

        for i in range(len(self.expected_output)):
            self.assertEqual(self.expected_output[i], predicted_result[i])

    def test_transform(self):
        n_gram = NGram()
        output = n_gram.transform(self.input_data_table)[0]
        self.verify_output_result(output)

    def test_save_load_transform(self):
        n_gram = NGram()
        path = os.path.join(self.temp_dir, 'test_save_load_transform_ngram')
        n_gram.save(path)
        n_gram = NGram.load(self.t_env, path)
        output = n_gram.transform(self.input_data_table)[0]
        self.verify_output_result(output)
