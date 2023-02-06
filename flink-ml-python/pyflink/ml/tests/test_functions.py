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
from pyflink.common import Types
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo, SparseVectorTypeInfo, \
    VectorTypeInfo
from pyflink.ml.functions import vector_to_array, array_to_vector
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase
from pyflink.table.expressions import col


class FunctionsTest(PyFlinkMLTestCase):
    def setUp(self):
        super(FunctionsTest, self).setUp()

        self.double_arrays = [
            ([0.0, 0.0],),
            ([0.0, 1.0],),
        ]

        self.float_arrays = [
            ([float(0.0), float(0.0)],),
            ([float(0.0), float(1.0)],),
        ]

        self.int_arrays = [
            ([0, 0],),
            ([0, 1],),
        ]

        self.dense_vectors = [
            (Vectors.dense(0.0, 0.0),),
            (Vectors.dense(0.0, 1.0),),
        ]

        self.sparse_vectors = [
            (Vectors.sparse(2, [], []),),
            (Vectors.sparse(2, [1], [1.0]),),
        ]

        self.mixed_vectors = [
            (Vectors.dense(0.0, 0.0),),
            (Vectors.sparse(2, [1], [1.0]),),
        ]

    def test_vector_to_array(self):
        self._test_vector_to_array(self.dense_vectors, DenseVectorTypeInfo())
        self._test_vector_to_array(self.sparse_vectors, SparseVectorTypeInfo())
        self._test_vector_to_array(self.mixed_vectors, VectorTypeInfo())

    def _test_vector_to_array(self, vectors, vector_type_info):
        input_table = self.t_env.from_data_stream(
            self.env.from_collection(vectors,
                                     type_info=Types.ROW_NAMED(
                                         ['vector'],
                                         [vector_type_info])
                                     ))

        output_table = input_table.select(vector_to_array(col('vector')).alias('array'))

        output_values = [x['array'] for x in self.t_env.to_data_stream(output_table)
                         .map(lambda r: r).execute_and_collect()]

        self.assertEqual(len(output_values), len(self.double_arrays))

        output_values.sort(key=lambda x: x[1])

        for i in range(len(self.double_arrays)):
            self.assertEqual(self.double_arrays[i][0], output_values[i])

    def test_array_to_vector(self):
        self._test_array_to_vector(self.double_arrays, Types.DOUBLE())
        self._test_array_to_vector(self.float_arrays, Types.FLOAT())
        self._test_array_to_vector(self.int_arrays, Types.INT())
        self._test_array_to_vector(self.int_arrays, Types.LONG())

    def _test_array_to_vector(self, arrays, array_element_type_info):
        input_table = self.t_env.from_data_stream(
            self.env.from_collection(
                arrays,
                type_info=Types.ROW_NAMED(
                    ['array'],
                    [Types.PRIMITIVE_ARRAY(array_element_type_info)]
                )
            )
        )

        output_table = input_table.select(array_to_vector(col('array')).alias('vector'))

        field_names = output_table.get_schema().get_field_names()

        output_values = [x[field_names.index('vector')] for x in
                         self.t_env.to_data_stream(output_table).execute_and_collect()]

        self.assertEqual(len(output_values), len(self.dense_vectors))

        output_values.sort(key=lambda x: x.get(1))

        for i in range(len(self.dense_vectors)):
            self.assertEqual(self.dense_vectors[i][0], output_values[i])
