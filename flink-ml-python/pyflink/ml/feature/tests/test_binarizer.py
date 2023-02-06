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

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo, SparseVectorTypeInfo
from pyflink.ml.feature.binarizer import Binarizer
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class BinarizerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(BinarizerTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (1,
                 Vectors.dense(1, 2),
                 Vectors.sparse(17, [0, 3, 9], [1.0, 2.0, 7.0])),
                (2,
                 Vectors.dense(2, 1),
                 Vectors.sparse(17, [0, 2, 14], [5.0, 4.0, 1.0])),
                (3,
                 Vectors.dense(5, 18),
                 Vectors.sparse(17, [0, 11, 12], [2.0, 4.0, 4.0]))
            ],
                type_info=Types.ROW_NAMED(
                    ['f0', 'f1', 'f2'],
                    [Types.INT(), DenseVectorTypeInfo(), SparseVectorTypeInfo()])))

        self.expected_output_data = [[0.0,
                                      Vectors.dense(0.0, 1.0),
                                      Vectors.sparse(17, [9], [1.0])],
                                     [1.0,
                                      Vectors.dense(1.0, 0.0),
                                      Vectors.sparse(17, [0, 2], [1.0, 1.0])],
                                     [1.0,
                                      Vectors.dense(1.0, 1.0),
                                      Vectors.sparse(17, [11, 12], [1.0, 1.0])]]

    def test_param(self):
        binarizer = Binarizer()

        binarizer.set_input_cols('f0', 'f1') \
            .set_output_cols('of0', 'of1') \
            .set_thresholds(1.5, 2.5)

        self.assertEqual(('f0', 'f1'), binarizer.input_cols)
        self.assertEqual(('of0', 'of1'), binarizer.output_cols)
        self.assertEqual((1.5, 2.5), binarizer.get_thresholds())

    def test_save_load_transform(self):
        binarizer = Binarizer() \
            .set_input_cols('f0', 'f1', 'f2') \
            .set_output_cols('of0', 'of1', 'of2') \
            .set_thresholds(1.0, 1.5, 2.5)

        path = os.path.join(self.temp_dir, 'test_save_load_transform_binarizer')
        binarizer.save(path)
        binarizer = Binarizer.load(self.t_env, path)

        output_table = binarizer.transform(self.input_data_table)[0]
        actual_outputs = [(result[0], result[3], result[4], result[5]) for result in
                          self.t_env.to_data_stream(output_table).execute_and_collect()]

        self.assertEqual(3, len(actual_outputs))

        actual_outputs.sort()

        for i in range(len(actual_outputs)):
            actual_output = actual_outputs[i]
            self.assertAlmostEqual(self.expected_output_data[i][0], actual_output[1], delta=1.0e-7)
            self.assertEqual(2, len(actual_output[2]))
            for j in range(len(actual_output[2])):
                self.assertAlmostEqual(self.expected_output_data[i][1].get(j),
                                       actual_output[2].get(j), delta=1e-7)
            self.assertEqual(17, len(actual_output[3]))
            for j in range(len(actual_output[3])):
                self.assertAlmostEqual(self.expected_output_data[i][2].get(j),
                                       actual_output[3].get(j), delta=1e-7)
