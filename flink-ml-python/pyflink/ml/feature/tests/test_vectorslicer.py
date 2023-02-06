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

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.vectorslicer import VectorSlicer
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class VectorSlicerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(VectorSlicerTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (1, Vectors.dense(2.1, 3.1, 1.2, 2.1)),
                (2, Vectors.dense(2.3, 2.1, 1.3, 1.2)),
            ],
                type_info=Types.ROW_NAMED(
                    ['id', 'vec'],
                    [Types.INT(), DenseVectorTypeInfo()])))

        self.expected_output_data_1 = Vectors.dense(2.1, 3.1, 1.2)
        self.expected_output_data_2 = Vectors.dense(2.3, 2.1, 1.3)

    def test_param(self):
        vector_slicer = VectorSlicer()

        self.assertEqual('input', vector_slicer.get_input_col())
        self.assertEqual('output', vector_slicer.get_output_col())

        vector_slicer.set_input_col('vec') \
            .set_output_col('slice_vec') \
            .set_indices(0, 1, 2)

        self.assertEqual('vec', vector_slicer.get_input_col())
        self.assertEqual((0, 1, 2), vector_slicer.get_indices())
        self.assertEqual('slice_vec', vector_slicer.get_output_col())

    def test_save_load_transform(self):
        vector_slicer = VectorSlicer() \
            .set_input_col('vec') \
            .set_output_col('slice_vec') \
            .set_indices(0, 1, 2)

        path = os.path.join(self.temp_dir, 'test_save_load_transform_vector_slicer')
        vector_slicer.save(path)
        vector_slicer = VectorSlicer.load(self.t_env, path)

        output_table = vector_slicer.transform(self.input_data_table)[0]
        actual_outputs = [(result[0], result[2]) for result in
                          self.t_env.to_data_stream(output_table).execute_and_collect()]

        self.assertEqual(2, len(actual_outputs))
        for actual_output in actual_outputs:
            if actual_output[0] == 1:
                self.assertEqual(self.expected_output_data_1, actual_output[1])
            else:
                self.assertEqual(self.expected_output_data_2, actual_output[1])
