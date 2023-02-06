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

from pyflink.ml.linalg import Vectors
from pyflink.ml.feature.hashingtf import HashingTF
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class HashingTFTest(PyFlinkMLTestCase):
    def setUp(self):
        super(HashingTFTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (['HashingTFTest', 'Hashing', 'Term', 'Frequency', 'Test'],),
                (['HashingTFTest', 'Hashing', 'Hashing', 'Test', 'Test'],),
            ],
                type_info=Types.ROW_NAMED(
                    ["input", ],
                    [Types.OBJECT_ARRAY(Types.STRING())])))

        self.expected_output = [
            Vectors.sparse(262144, [67564, 89917, 113827, 131486, 228971],
                           [1.0, 1.0, 1.0, 1.0, 1.0]),
            Vectors.sparse(262144, [67564, 131486, 228971], [1.0, 2.0, 2.0])
        ]

        self.expected_binary_output = [
            Vectors.sparse(262144, [67564, 89917, 113827, 131486, 228971],
                           [1.0, 1.0, 1.0, 1.0, 1.0]),
            Vectors.sparse(262144, [67564, 131486, 228971], [1.0, 1.0, 1.0])
        ]

    def test_param(self):
        hashing_tf = HashingTF()
        self.assertEqual('input', hashing_tf.input_col)
        self.assertFalse(hashing_tf.binary)
        self.assertEqual(262144, hashing_tf.num_features)
        self.assertEqual('output', hashing_tf.output_col)

        hashing_tf.set_input_col("test_input_col") \
            .set_binary(True) \
            .set_num_features(1024) \
            .set_output_col("test_output_col")

        self.assertEqual('test_input_col', hashing_tf.input_col)
        self.assertTrue(hashing_tf.binary)
        self.assertEqual(1024, hashing_tf.num_features)
        self.assertEqual('test_output_col', hashing_tf.output_col)

    def test_output_schema(self):
        hashing_tf = HashingTF()
        input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ([''], ''),
            ],
                type_info=Types.ROW_NAMED(
                    ['input', 'dummy_input'],
                    [Types.OBJECT_ARRAY(Types.STRING()), Types.STRING()])))

        output = hashing_tf \
            .set_input_col('input') \
            .set_output_col('output') \
            .transform(input_data_table)[0]

        self.assertEqual(
            [hashing_tf.input_col, 'dummy_input', hashing_tf.output_col],
            output.get_schema().get_field_names())

    def verify_output_result(self, output_table, expected_output):
        predicted_result = [result[1] for result in
                            self.t_env.to_data_stream(output_table).execute_and_collect()]
        expected_output.sort(key=lambda x: x[89917])
        predicted_result.sort(key=lambda x: x[89917])
        self.assertEqual(len(expected_output), len(predicted_result))

        for i in range(len(expected_output)):
            self.assertEqual(expected_output[i], predicted_result[i])

    def test_transform(self):
        hashing_tf = HashingTF()

        # Tests non-binary.
        output = hashing_tf.transform(self.input_data_table)[0]
        self.verify_output_result(output, self.expected_output)

        # Tests binary.
        hashing_tf.set_binary(True)
        output = hashing_tf.transform(self.input_data_table)[0]
        self.verify_output_result(output, self.expected_binary_output)

    def test_save_load_transform(self):
        hashingtf = HashingTF()
        path = os.path.join(self.temp_dir, 'test_save_load_transform_hashingtf')
        hashingtf.save(path)
        hashingtf = HashingTF.load(self.t_env, path)
        output = hashingtf.transform(self.input_data_table)[0]
        self.verify_output_result(output, self.expected_output)
