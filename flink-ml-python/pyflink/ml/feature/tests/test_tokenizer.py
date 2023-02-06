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

from pyflink.ml.feature.tokenizer import Tokenizer
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class TokenizerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(TokenizerTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ('Test for tokenization.',),
                ('Te,st. punct',),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [Types.STRING()])))
        self.expected_output = [
            ['test', 'for', 'tokenization.'],
            ['te,st.', 'punct']
        ]

    def test_param(self):
        tokenizer = Tokenizer()
        self.assertEqual('input', tokenizer.input_col)
        self.assertEqual('output', tokenizer.output_col)

        tokenizer.set_input_col('testInputCol').set_output_col('testOutputCol')
        self.assertEqual('testInputCol', tokenizer.input_col)
        self.assertEqual('testOutputCol', tokenizer.output_col)

    def test_output_schema(self):
        tokenizer = Tokenizer()
        input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ('', ''),
            ],
                type_info=Types.ROW_NAMED(
                    ['input', 'dummy_input'],
                    [Types.STRING(), Types.STRING()])))
        output = tokenizer.transform(input_data_table)[0]

        self.assertEqual(
            [tokenizer.input_col, 'dummy_input', tokenizer.output_col],
            output.get_schema().get_field_names())

    def test_save_load_transform(self):
        tokenizer = Tokenizer()
        path = os.path.join(self.temp_dir, 'test_save_load_transform_tokenizer')
        tokenizer.save(path)
        tokenizer = Tokenizer.load(self.t_env, path)
        output_table = tokenizer.transform(self.input_data_table)[0]
        predicted_results = [result[1] for result in
                             self.t_env.to_data_stream(output_table).execute_and_collect()]

        predicted_results.sort(key=lambda x: x[0])
        self.expected_output.sort(key=lambda x: x[0])
        self.assertEqual(self.expected_output, predicted_results)
