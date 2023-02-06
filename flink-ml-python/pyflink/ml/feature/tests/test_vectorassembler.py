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
from pyflink.ml.feature.vectorassembler import VectorAssembler
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class VectorAssemblerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(VectorAssemblerTest, self).setUp()
        # TODO: Add test for handling invalid values after FLINK-27797 is resolved.
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (0,
                 Vectors.dense(2.1, 3.1),
                 1.0,
                 Vectors.sparse(5, [3], [1.0])),
                (1,
                 Vectors.dense(2.1, 3.1),
                 1.0,
                 Vectors.sparse(5, [1, 2, 3, 4],
                                [1.0, 2.0, 3.0, 4.0])),
            ],
                type_info=Types.ROW_NAMED(
                    ['id', 'vec', 'num', 'sparse_vec'],
                    [Types.INT(), DenseVectorTypeInfo(), Types.DOUBLE(), SparseVectorTypeInfo()])))

        self.expected_output_data_1 = Vectors.sparse(8, [0, 1, 2, 6], [2.1, 3.1, 1.0, 1.0])
        self.expected_output_data_2 = Vectors.dense(2.1, 3.1, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0)

    def test_param(self):
        vector_assembler = VectorAssembler()

        self.assertEqual('error', vector_assembler.handle_invalid)
        self.assertEqual('output', vector_assembler.output_col)

        vector_assembler.set_input_cols('vec', 'num', 'sparse_vec') \
            .set_output_col('assembled_vec') \
            .set_input_sizes(2, 1, 5) \
            .set_handle_invalid('skip')

        self.assertEqual(('vec', 'num', 'sparse_vec'), vector_assembler.input_cols)
        self.assertEqual('assembled_vec', vector_assembler.output_col)
        self.assertEqual((2, 1, 5), vector_assembler.input_sizes)
        self.assertEqual('skip', vector_assembler.handle_invalid)

    def test_save_load_transform(self):
        vector_assembler = VectorAssembler() \
            .set_input_cols('vec', 'num', 'sparse_vec') \
            .set_output_col('assembled_vec') \
            .set_input_sizes(2, 1, 5) \
            .set_handle_invalid('keep')

        path = os.path.join(self.temp_dir, 'test_save_load_transform_vector_assembler')
        vector_assembler.save(path)
        vector_assembler = VectorAssembler.load(self.t_env, path)

        output_table = vector_assembler.transform(self.input_data_table)[0]
        actual_outputs = [(result[0], result[4]) for result in
                          self.t_env.to_data_stream(output_table).execute_and_collect()]

        for actual_output in actual_outputs:
            if actual_output[0] == 0:
                self.assertEqual(self.expected_output_data_1, actual_output[1])
            else:
                self.assertEqual(self.expected_output_data_2, actual_output[1])
