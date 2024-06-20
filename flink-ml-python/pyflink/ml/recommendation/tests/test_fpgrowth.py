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
from pyflink.table import Table
from typing import List

from pyflink.ml.recommendation.fpgrowth import FPGrowth
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


# Tests Swing.
class SwingTest(PyFlinkMLTestCase):
    def setUp(self):
        super(SwingTest, self).setUp()
        self.input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ("A,B,C,D",),
                ("B,C,E",),
                ("A,B,C,E",),
                ("B,D,E",),
                ("A,B,C,D",)
            ],
                type_info=Types.ROW_NAMED(
                    ['items'],
                    [Types.STRING()])
            ))

        self.expected_patterns = [
            ["A", 3, 1],
            ["B", 5, 1],
            ["B,A", 3, 2],
            ["B,C", 4, 2],
            ["B,C,A", 3, 3],
            ["B,D", 3, 2],
            ["B,E", 3, 2],
            ["C", 4, 1],
            ["C,A", 3, 2],
            ["D", 3, 1],
            ["E", 3, 1]
        ]

        self.expected_rules = [
            ["A=>B", 2, 1.0, 0.6, 1.0, 3],
            ["A=>C", 2, 1.25, 0.6, 1.0, 3],
            ["B=>A", 2, 1.0, 0.6, 0.6, 3],
            ["B=>C", 2, 1.0, 0.8, 0.8, 4],
            ["B=>D", 2, 1.0, 0.6, 0.6, 3],
            ["B=>E", 2, 1.0, 0.6, 0.6, 3],
            ["B,A=>C", 3, 1.25, 0.6, 1.0, 3],
            ["B,C=>A", 3, 1.25, 0.6, 0.75, 3],
            ["C=>A", 2, 1.25, 0.6, 0.75, 3],
            ["C=>B", 2, 1.0, 0.8, 1.0, 4],
            ["C,A=>B", 3, 1.0, 0.6, 1.0, 3],
            ["D=>B", 2, 1.0, 0.6, 1.0, 3],
            ["E=>B", 2, 1.0, 0.6, 1.0, 3]
        ]

    def test_param(self):
        fpg = FPGrowth()
        self.assertEqual("items", fpg.items_col)
        self.assertEqual(",", fpg.field_delimiter)
        self.assertAlmostEqual(1.0, fpg.min_lift, delta=1e-9)
        self.assertAlmostEqual(0.6, fpg.min_confidence, delta=1e-9)
        self.assertAlmostEqual(0.02, fpg.min_support, delta=1e-9)
        self.assertEqual(-1, fpg.min_support_count)
        self.assertEqual(10, fpg.max_pattern_length)

        fpg.set_items_col("values") \
            .set_field_delimiter(" ") \
            .set_min_lift(1.2) \
            .set_min_confidence(0.7) \
            .set_min_support(0.01) \
            .set_min_support_count(50) \
            .set_max_pattern_length(5)

        self.assertEqual("values", fpg.items_col)
        self.assertEqual(" ", fpg.field_delimiter)
        self.assertAlmostEqual(1.2, fpg.min_lift, delta=1e-9)
        self.assertAlmostEqual(0.7, fpg.min_confidence, delta=1e-9)
        self.assertAlmostEqual(0.01, fpg.min_support, delta=1e-9)
        self.assertEqual(50, fpg.min_support_count)
        self.assertEqual(5, fpg.max_pattern_length)

    def test_output_schema(self):
        fpg = FPGrowth()
        output_tables = fpg.transform(self.input_table)
        self.assertEqual(
            ["items", "support_count", "item_count"],
            output_tables[0].get_schema().get_field_names())
        self.assertEqual(
            ["rule", "item_count", "lift", "support_percent",
             "confidence_percent", "transaction_count"],
            output_tables[1].get_schema().get_field_names())

    def test_transform(self):
        fpg = FPGrowth().set_min_support(0.6)
        output_tables = fpg.transform(self.input_table)
        self.verify_output_result(output_tables[0], self.expected_patterns)
        self.verify_output_result(output_tables[1], self.expected_rules)

    def test_save_load_and_transform(self):
        fpg = FPGrowth().set_min_support_count(3)
        reloaded_swing = self.save_and_reload(fpg)
        output_tables = reloaded_swing.transform(self.input_table)
        self.verify_output_result(output_tables[0], self.expected_patterns)
        self.verify_output_result(output_tables[1], self.expected_rules)

    def verify_output_result(
            self, output: Table,
            expected_result: List):
        collected_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]
        results = []
        for result in collected_results:
            results.append([item for item in result])
        results.sort(key=lambda x: x[0])
        expected_result.sort(key=lambda x: x[0])
        self.assertEqual(expected_result, results)
