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

from pyflink.ml.feature.regextokenizer import RegexTokenizer
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class RegexTokenizerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(RegexTokenizerTest, self).setUp()
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
        regex_tokenizer = RegexTokenizer()
        self.assertEqual('input', regex_tokenizer.input_col)
        self.assertEqual('output', regex_tokenizer.output_col)
        self.assertEqual(1, regex_tokenizer.min_token_length)
        self.assertEqual(True, regex_tokenizer.gaps)
        self.assertEqual('\\s+', regex_tokenizer.pattern)
        self.assertEqual(True, regex_tokenizer.to_lowercase)

        regex_tokenizer \
            .set_input_col("testInputCol") \
            .set_output_col("testOutputCol") \
            .set_min_token_length(3) \
            .set_gaps(False) \
            .set_pattern("\\s") \
            .set_to_lowercase(False)

        self.assertEqual('testInputCol', regex_tokenizer.input_col)
        self.assertEqual('testOutputCol', regex_tokenizer.output_col)
        self.assertEqual(3, regex_tokenizer.min_token_length)
        self.assertEqual(False, regex_tokenizer.gaps)
        self.assertEqual('\\s', regex_tokenizer.pattern)
        self.assertEqual(False, regex_tokenizer.to_lowercase)

    def test_output_schema(self):
        regex_tokenizer = RegexTokenizer()
        input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ('', ''),
            ],
                type_info=Types.ROW_NAMED(
                    ['input', 'dummy_input'],
                    [Types.STRING(), Types.STRING()])))
        output = regex_tokenizer.transform(input_data_table)[0]

        self.assertEqual(
            [regex_tokenizer.input_col, 'dummy_input', regex_tokenizer.output_col],
            output.get_schema().get_field_names())

    def test_save_load_transform(self):
        regex_tokenizer = RegexTokenizer()
        path = os.path.join(self.temp_dir, 'test_save_load_transform_regextokenizer')
        regex_tokenizer.save(path)
        regex_tokenizer = RegexTokenizer.load(self.t_env, path)
        output_table = regex_tokenizer.transform(self.input_data_table)[0]
        predicted_results = [result[1] for result in
                             self.t_env.to_data_stream(output_table).execute_and_collect()]

        predicted_results.sort(key=lambda x: x[0])
        self.expected_output.sort(key=lambda x: x[0])
        self.assertEqual(self.expected_output, predicted_results)
