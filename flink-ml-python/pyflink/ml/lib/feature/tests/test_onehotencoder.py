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

from typing import List, Dict, Tuple

from pyflink.common import Types
from pyflink.table import Table

from pyflink.ml.core.linalg import Vectors, SparseVector
from pyflink.ml.lib.feature.onehotencoder import OneHotEncoder, OneHotEncoderModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class OneHotEncoderTest(PyFlinkMLTestCase):
    def setUp(self):
        super(OneHotEncoderTest, self).setUp()
        self.train_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (0.0,),
                (1.0,),
                (2.0,),
                (0.0,),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [Types.DOUBLE()])))

        self.predict_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (0.0,),
                (1.0,),
                (2.0,),
            ],
                type_info=Types.ROW_NAMED(
                    ['input'],
                    [Types.DOUBLE()])))
        self.expected_data = {
            0.0: Vectors.sparse(2, [0], [1.0]),
            1.0: Vectors.sparse(2, [1], [1.0]),
            2.0: Vectors.sparse(2, [], [])
        }

        self.estimator = OneHotEncoder().set_input_cols('input').set_output_cols('output')

    def test_param(self):
        estimator = OneHotEncoder()

        self.assertTrue(estimator.drop_last)

        estimator.set_input_cols('test_input') \
            .set_output_cols('test_output') \
            .set_drop_last(False)

        self.assertEqual(('test_input',), estimator.input_cols)
        self.assertEqual(('test_output',), estimator.output_cols)
        self.assertFalse(estimator.drop_last)

        model = OneHotEncoderModel()

        model.set_input_cols('test_input').set_output_cols('test_output').set_drop_last(False)

        self.assertEqual(('test_input',), model.input_cols)
        self.assertEqual(('test_output',), model.output_cols)
        self.assertFalse(model.drop_last)

    def test_fit_and_predict(self):
        model = self.estimator.fit(self.train_data)  # type: OneHotEncoderModel
        output_table = model.transform(self.predict_data)[0]
        self.verify_output_result(
            output_table,
            model.input_cols,
            model.output_cols,
            output_table.get_schema().get_field_names(),
            self.expected_data)

    def test_drop_last(self):
        self.estimator.set_drop_last(False)

        expected_data = {
            0.0: Vectors.sparse(3, [0], [1.0]),
            1.0: Vectors.sparse(3, [1], [1.0]),
            2.0: Vectors.sparse(3, [2], [1.0])
        }

        model = self.estimator.fit(self.train_data)  # type: OneHotEncoderModel
        output_table = model.transform(self.predict_data)[0]
        self.verify_output_result(
            output_table,
            model.input_cols,
            model.output_cols,
            output_table.get_schema().get_field_names(),
            expected_data)

    def verify_output_result(
            self,
            output: Table,
            input_cols: Tuple[str],
            output_cols: Tuple[str],
            field_names: List[str],
            expected_result: Dict[float, SparseVector]):
        collected_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]

        results = {}
        for item in collected_results:
            item.set_field_names(field_names)
            for i in range(len(input_cols)):
                results[item[input_cols[i]]] = item[output_cols[i]]
        self.assertEqual(expected_result, results)
