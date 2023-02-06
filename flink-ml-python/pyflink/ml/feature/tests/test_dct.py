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
from typing import List

from pyflink.common import Row, Types
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.dct import DCT
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase
from pyflink.table import Table


class DCTTest(PyFlinkMLTestCase):
    def setUp(self):
        super(DCTTest, self).setUp()
        self.input = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(1.0, 1.0, 1.0, 1.0),),
                (Vectors.dense(1.0, 0.0, -1.0, 0.0),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [DenseVectorTypeInfo()])))

        self.expected_output = [
            Row(Vectors.dense(1.0, 1.0, 1.0, 1.0), Vectors.dense(2.0, 0.0, 0.0, 0.0)),
            Row(Vectors.dense(1.0, 0.0, -1.0, 0.0), Vectors.dense(0.0, 0.924, 1.0, -0.383))]

    def test_param(self):
        dct = DCT()

        self.assertEqual('input', dct.get_input_col())
        self.assertEqual('output', dct.get_output_col())
        self.assertFalse(dct.get_inverse())

        dct.set_input_col('test_input') \
            .set_output_col('test_output') \
            .set_inverse(True)

        self.assertEqual('test_input', dct.get_input_col())
        self.assertEqual('test_output', dct.get_output_col())
        self.assertTrue(dct.get_inverse())

    def test_output_schema(self):
        temp_table = self.input.alias('test_input')
        dct = DCT().set_input_col('test_input').set_output_col('test_output')
        output = dct.transform(temp_table)[0]
        self.assertEqual(['test_input', 'test_output'], output.get_schema().get_field_names())

    def test_transform(self):
        dct = DCT()
        output = dct.transform(self.input)[0]
        self.verify_output_result(output, self.expected_output)

    def test_save_load_transform(self):
        dct = DCT()
        path = os.path.join(self.temp_dir, 'test_save_load_transform_dct')
        dct.save(path)
        dct = DCT.load(self.t_env, path)
        output = dct.transform(self.input)[0]
        self.verify_output_result(output, self.expected_output)

    def verify_output_result(
            self,
            output: Table,
            expected_result: List[Row]):
        actual_result = [result for result in
                         self.t_env.to_data_stream(output).execute_and_collect()]
        actual_result.sort(key=lambda x: hash(x[0]))
        expected_result.sort(key=lambda x: hash(x[0]))

        for item1, item2 in zip(actual_result, expected_result):
            for i, j in zip(item1[1]._values, item2[1]._values):
                self.assertAlmostEqual(i, j, delta=1e-3)
