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
from pyflink.ml.feature.polynomialexpansion import PolynomialExpansion
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class PolynomialExpansionTest(PyFlinkMLTestCase):
    def setUp(self):
        super(PolynomialExpansionTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(1.0, 2.0),),
                (Vectors.dense(2.0, 3.0),),
            ],
                type_info=Types.ROW_NAMED(
                    ["intput_vec"],
                    [DenseVectorTypeInfo()])))
        self.expected_output_data = [
            Vectors.dense(1.0, 1.0, 2.0, 2.0, 4.0),
            Vectors.dense(2.0, 4.0, 3.0, 6.0, 9.0)]

    def test_param(self):
        polynomialexpansion = PolynomialExpansion()

        self.assertEqual('input', polynomialexpansion.get_input_col())
        self.assertEqual('output', polynomialexpansion.get_output_col())
        self.assertEqual(2, polynomialexpansion.get_degree())

        polynomialexpansion.set_input_col("intput_vec") \
            .set_output_col('output_vec') \
            .set_degree(3)

        self.assertEqual("intput_vec", polynomialexpansion.get_input_col())
        self.assertEqual(3, polynomialexpansion.get_degree())
        self.assertEqual('output_vec', polynomialexpansion.get_output_col())

    def test_save_load_transform(self):
        polynomialexpansion = PolynomialExpansion() \
            .set_input_col("intput_vec") \
            .set_output_col('output_vec') \
            .set_degree(2)

        path = os.path.join(self.temp_dir, 'test_save_load_transform_polynomialexpansion')
        polynomialexpansion.save(path)
        polynomialexpansion = PolynomialExpansion.load(self.t_env, path)

        output_table = polynomialexpansion.transform(self.input_data_table)[0]
        actual_outputs = [(result[1]) for result in
                          self.t_env.to_data_stream(output_table).execute_and_collect()]

        self.assertEqual(2, len(actual_outputs))
        actual_outputs.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        self.expected_output_data.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        self.assertEqual(self.expected_output_data, actual_outputs)
