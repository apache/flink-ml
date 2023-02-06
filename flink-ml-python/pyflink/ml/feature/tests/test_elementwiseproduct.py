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
from pyflink.ml.feature.elementwiseproduct import ElementwiseProduct
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class ElementwiseProductTest(PyFlinkMLTestCase):
    def setUp(self):
        super(ElementwiseProductTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (0,
                 Vectors.dense(2.1, 3.1)),
                (1,
                 Vectors.dense(1.1, 3.3)),
            ],
                type_info=Types.ROW_NAMED(
                    ['id', 'vec'],
                    [Types.INT(), DenseVectorTypeInfo()])))

        self.expected_output_data_1 = Vectors.dense(2.31, 3.41)
        self.expected_output_data_2 = Vectors.dense(1.21, 3.63)

    def test_param(self):
        elementwise_product = ElementwiseProduct()

        self.assertEqual('input', elementwise_product.get_input_col())
        self.assertEqual('output', elementwise_product.get_output_col())

        elementwise_product.set_input_col('vec') \
            .set_output_col('output_vec') \
            .set_scaling_vec(Vectors.dense(1.1, 1.1))

        self.assertEqual('vec', elementwise_product.get_input_col())
        self.assertEqual(Vectors.dense(1.1, 1.1), elementwise_product.get_scaling_vec())
        self.assertEqual('output_vec', elementwise_product.get_output_col())

    def test_save_load_transform(self):
        elementwise_product = ElementwiseProduct() \
            .set_input_col('vec') \
            .set_output_col('output_vec') \
            .set_scaling_vec(Vectors.dense(1.1, 1.1))

        path = os.path.join(self.temp_dir, 'test_save_load_transform_elementwise_product')
        elementwise_product.save(path)
        elementwise_product = ElementwiseProduct.load(self.t_env, path)

        output_table = elementwise_product.transform(self.input_data_table)[0]
        actual_outputs = [(result[0], result[2]) for result in
                          self.t_env.to_data_stream(output_table).execute_and_collect()]

        self.assertEqual(2, len(actual_outputs))
        for actual_output in actual_outputs:
            self.assertEqual(2, len(actual_output[1]))
            if actual_output[0] == 0:
                for i in range(len(actual_output[1])):
                    self.assertAlmostEqual(self.expected_output_data_1.get(i),
                                           actual_output[1].get(i), delta=1e-7)
            else:
                for i in range(len(actual_output[1])):
                    self.assertAlmostEqual(self.expected_output_data_2.get(i),
                                           actual_output[1].get(i), delta=1e-7)
