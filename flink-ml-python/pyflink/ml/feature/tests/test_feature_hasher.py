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
from pyflink.ml.feature.featurehasher import FeatureHasher
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class FeatureHasherTest(PyFlinkMLTestCase):
    def setUp(self):
        super(FeatureHasherTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (0, 'a', 1.0, True),
                (1, 'c', 1.0, False)
            ],
                type_info=Types.ROW_NAMED(
                    ['id', 'f0', 'f1', 'f2'],
                    [Types.INT(), Types.STRING(), Types.DOUBLE(), Types.BOOLEAN()])))

        self.expected_output_data_1 = Vectors.sparse(1000, [607, 635, 913], [1.0, 1.0, 1.0])
        self.expected_output_data_2 = Vectors.sparse(1000, [242, 869, 913], [1.0, 1.0, 1.0])

    def test_param(self):
        feature_hasher = FeatureHasher()

        self.assertEqual('output', feature_hasher.output_col)
        self.assertEqual(262144, feature_hasher.num_features)

        feature_hasher.set_input_cols('f0', 'f1', 'f2') \
            .set_categorical_cols('f0', 'f2') \
            .set_output_col('vec') \
            .set_num_features(1000)

        self.assertEqual(('f0', 'f1', 'f2'), feature_hasher.input_cols)
        self.assertEqual(('f0', 'f2'), feature_hasher.categorical_cols)
        self.assertEqual(1000, feature_hasher.num_features)
        self.assertEqual('vec', feature_hasher.output_col)

    def test_save_load_transform(self):
        feature_hasher = FeatureHasher() \
            .set_input_cols('f0', 'f1', 'f2') \
            .set_categorical_cols('f0', 'f2') \
            .set_output_col('vec') \
            .set_num_features(1000)

        path = os.path.join(self.temp_dir, 'test_save_load_transform_feature_hasher')
        feature_hasher.save(path)
        feature_hasher = FeatureHasher.load(self.t_env, path)

        output_table = feature_hasher.transform(self.input_data_table)[0]
        actual_outputs = [(result[0], result[4]) for result in
                          self.t_env.to_data_stream(output_table).execute_and_collect()]
        self.assertEqual(2, len(actual_outputs))
        for actual_output in actual_outputs:
            if actual_output[0] == 0:
                self.assertEqual(self.expected_output_data_1, actual_output[1])
            else:
                self.assertEqual(self.expected_output_data_2, actual_output[1])
