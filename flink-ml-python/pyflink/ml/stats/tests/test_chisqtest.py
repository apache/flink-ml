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

from pyflink.common import Types, Row
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo

from pyflink.ml.stats.chisqtest import ChiSqTest


class ChiSqTestTest(PyFlinkMLTestCase):
    def setUp(self):
        super(ChiSqTestTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (0., Vectors.dense(5, 1.)),
                (2., Vectors.dense(6, 2.)),
                (1., Vectors.dense(7, 2.)),
                (1., Vectors.dense(5, 4.)),
                (0., Vectors.dense(5, 1.)),
                (2., Vectors.dense(6, 2.)),
                (1., Vectors.dense(7, 2.)),
                (1., Vectors.dense(5, 4.)),
                (2., Vectors.dense(5, 1.)),
                (0., Vectors.dense(5, 2.)),
                (0., Vectors.dense(5, 2.)),
                (1., Vectors.dense(9, 4.)),
                (1., Vectors.dense(9, 3.))
            ],
                type_info=Types.ROW_NAMED(
                    ['label', 'features'],
                    [Types.DOUBLE(), DenseVectorTypeInfo()]))
        )

        self.expected_output_data = [
            Row(0, 0.03419350755, 6, 13.61904761905),
            Row(1, 0.24220177737, 6, 7.94444444444)]

    def test_param(self):
        chi_sq_test = ChiSqTest()
        self.assertEqual("label", chi_sq_test.label_col)
        self.assertEqual("features", chi_sq_test.features_col)
        self.assertFalse(chi_sq_test.flatten)

        chi_sq_test.set_label_col("test_label") \
            .set_features_col("test_features") \
            .set_flatten(True)

        self.assertEqual("test_label", chi_sq_test.label_col)
        self.assertEqual("test_features", chi_sq_test.features_col)
        self.assertTrue(chi_sq_test.flatten)

    def test_output_schema(self):
        chi_sq_test = ChiSqTest()

        output = chi_sq_test.transform(self.input_data_table)[0]
        self.assertEqual(
            ["pValues",
             "degreesOfFreedom",
             "statistics"],
            output.get_schema().get_field_names())

        chi_sq_test.set_flatten(True)

        output = chi_sq_test.transform(self.input_data_table)[0]
        self.assertEqual(
            ["featureIndex",
             "pValue",
             "degreeOfFreedom",
             "statistic"],
            output.get_schema().get_field_names())

    def test_transform(self):
        # TODO: Add test for non-flatten case after upgrading Flink dependency to
        # 1.15.3, 1.16.1 or 1.17.0. Related jira: FLINK-29477
        chi_sq_test = ChiSqTest().set_flatten(True)
        output = chi_sq_test.transform(self.input_data_table)[0]
        actual_output_data = [
            result for result in self.t_env.to_data_stream(output).execute_and_collect()
        ]
        self.assertEqual(self.expected_output_data, actual_output_data)

    def test_save_load_and_transform(self):
        chi_sq_test = ChiSqTest().set_flatten(True)
        path = os.path.join(self.temp_dir, 'test_save_load_and_transform_chisqtest')
        chi_sq_test.save(path)
        chi_sq_test = ChiSqTest.load(self.t_env, path)
        output = chi_sq_test.transform(self.input_data_table)[0]
        actual_output_data = [
            result for result in self.t_env.to_data_stream(output).execute_and_collect()
        ]
        self.assertEqual(self.expected_output_data, actual_output_data)
