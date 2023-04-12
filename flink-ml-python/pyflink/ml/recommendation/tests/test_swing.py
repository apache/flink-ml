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

from pyflink.ml.recommendation.swing import Swing
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


# Tests Swing.
class SwingTest(PyFlinkMLTestCase):
    def setUp(self):
        super(SwingTest, self).setUp()
        self.input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (0, 10),
                (0, 11),
                (0, 12),
                (1, 13),
                (1, 12),
                (2, 10),
                (2, 11),
                (2, 12),
                (3, 13),
                (3, 12)
            ],
                type_info=Types.ROW_NAMED(
                    ['user', 'item'],
                    [Types.LONG(), Types.LONG()])
            ))

        self.wrong_type_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (0, 10),
                (1, 11),
                (2, 12)
            ],
                type_info=Types.ROW_NAMED(
                    ['user', 'item'],
                    [Types.INT(), Types.LONG()])
            ))

        self.none_value_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (0, 10),
                (None, 11),
                (2, 12)
            ],
                type_info=Types.ROW_NAMED(
                    ['user', 'item'],
                    [Types.LONG(), Types.LONG()])
            ))

        self.expected_data = [
            [10, '11,0.058845768947156235;12,0.058845768947156235'],
            [11, '10,0.058845768947156235;12,0.058845768947156235'],
            [12, '13,0.09134833828228624;10,0.058845768947156235;11,0.058845768947156235'],
            [13, '12,0.09134833828228624']]

    def test_param(self):
        swing = Swing()
        self.assertEqual("item", swing.item_col)
        self.assertEqual("user", swing.user_col)
        self.assertEqual(100, swing.k)
        self.assertEqual(1000, swing.max_user_num_per_item)
        self.assertEqual(10, swing.min_user_behavior)
        self.assertEqual(1000, swing.max_user_behavior)
        self.assertEqual(15, swing.alpha1)
        self.assertEqual(0, swing.alpha2)
        self.assertAlmostEqual(0.3, swing.beta, delta=1e-9)
        self.assertEqual(438758276, swing.seed)

        swing.set_item_col("item_1") \
            .set_user_col("user_1") \
            .set_k(20) \
            .set_max_user_num_per_item(500) \
            .set_min_user_behavior(20) \
            .set_max_user_behavior(50) \
            .set_alpha1(5) \
            .set_alpha2(1) \
            .set_beta(0.35) \
            .set_seed(1)

        self.assertEqual("item_1", swing.item_col)
        self.assertEqual("user_1", swing.user_col)
        self.assertEqual(20, swing.k)
        self.assertEqual(500, swing.max_user_num_per_item)
        self.assertEqual(20, swing.min_user_behavior)
        self.assertEqual(50, swing.max_user_behavior)
        self.assertEqual(5, swing.alpha1)
        self.assertEqual(1, swing.alpha2)
        self.assertAlmostEqual(0.35, swing.beta, delta=1e-9)
        self.assertEqual(1, swing.seed)

    def test_output_schema(self):
        swing = Swing() \
            .set_item_col('test_item') \
            .set_user_col('test_user') \
            .set_output_col("item_score")

        output = swing.transform(self.input_table.alias('test_user', 'test_item'))[0]
        self.assertEqual(
            ['test_item', 'item_score'],
            output.get_schema().get_field_names())

    def test_transform(self):
        swing = Swing().set_min_user_behavior(1)
        output = swing.transform(self.input_table)[0]
        self.verify_output_result(
            output,
            swing.get_item_col(),
            output.get_schema().get_field_names(),
            self.expected_data)

    def test_save_load_and_transform(self):
        swing = Swing().set_min_user_behavior(1)
        reloaded_swing = self.save_and_reload(swing)
        output = reloaded_swing.transform(self.input_table)[0]
        self.verify_output_result(
            output,
            swing.get_item_col(),
            output.get_schema().get_field_names(),
            self.expected_data)

    def verify_output_result(
            self, output: Table,
            item_col: str,
            field_names: List[str],
            expected_result: List):
        collected_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]
        results = []
        for result in collected_results:
            main_item = result[field_names.index(item_col)]
            item_rank_score = result[1]
            results.append([main_item, item_rank_score])
        results.sort(key=lambda x: x[0])
        self.assertEqual(expected_result, results)

    def test_sampling_method(self):
        swing1 = Swing().set_min_user_behavior(1).set_max_user_num_per_item(2).set_seed(3)
        swing2 = Swing().set_min_user_behavior(1).set_max_user_num_per_item(2)
        output1 = swing1.transform(self.input_table)[0]
        output2 = swing2.transform(self.input_table)[0]
        result1 = [result for result in self.t_env.to_data_stream(output1).execute_and_collect()]
        result2 = [result for result in self.t_env.to_data_stream(output2).execute_and_collect()]
        self.assertNotEqual(len(result1), len(result2))
