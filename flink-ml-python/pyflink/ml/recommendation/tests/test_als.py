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

from pyflink.ml.recommendation.als import Als
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


# Tests Als.
class AlsTest(PyFlinkMLTestCase):
    def setUp(self):
        super(AlsTest, self).setUp()
        self.input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (1, 5, 0.1),
                (2, 8, 0.5),
                (3, 5, 0.8),
                (4, 7, 0.1),
                (1, 7, 0.7),
                (2, 5, 0.9),
                (3, 8, 0.1),
                (2, 6, 0.7),
                (2, 7, 0.4),
                (1, 8, 0.3),
                (4, 6, 0.4),
                (3, 7, 0.6),
                (1, 6, 0.5),
                (4, 8, 0.3)
            ],
                type_info=Types.ROW_NAMED(
                    ['user', 'item', 'rating'],
                    [Types.LONG(), Types.LONG(), Types.DOUBLE()])
            ))

        self.test_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (1, 6)
            ],
                type_info=Types.ROW_NAMED(
                    ['user', 'item'],
                    [Types.LONG(), Types.LONG()])
            ))

    def test_param(self):
        als = Als()
        self.assertEqual('item', als.item_col)
        self.assertEqual('user', als.user_col)
        self.assertEqual('rating', als.rating_col)
        self.assertEqual(10, als.rank)
        self.assertEqual(10, als.max_iter)
        self.assertEqual(False, als.non_negative)
        self.assertEqual(False, als.implicit_refs)
        self.assertEqual(1.0, als.alpha)
        self.assertEqual(0.1, als.reg_param)
        self.assertEqual(als.prediction_col, 'prediction')

        als.set_item_col('item_1') \
            .set_user_col('user_1') \
            .set_rating_col('rating_1') \
            .set_rank(50) \
            .set_max_iter(30) \
            .set_non_negative(True) \
            .set_implicit_refs(True) \
            .set_alpha(0.35) \
            .set_reg_param(0.25) \
            .set_prediction_col('prediction_col')

        self.assertEqual('item_1', Als.item_col)
        self.assertEqual('user_1', Als.user_col)
        self.assertEqual('rating_1', als.rating_col)
        self.assertEqual(50, als.rank)
        self.assertEqual(30, als.max_iter)
        self.assertEqual(True, als.non_negative)
        self.assertEqual(True, als.implicit_refs)
        self.assertEqual(0.35, als.alpha)
        self.assertEqual(0.25, als.reg_param)
        self.assertEqual(als.prediction_col, 'prediction_col')

    def test_output_schema(self):
        als = Als() \
            .set_item_col('test_item') \
            .set_user_col('test_user') \
            .set_rating_col('test_rating') \
            .set_prediction_col('prediction_col')
        output = als.transform(self.input_table.alias('test_user', 'test_item', 'test_rating'))[0]
        self.assertEqual(
            ['test_user', 'test_item', 'test_rating', 'prediction_col'],
            output.get_schema().get_field_names())

    def test_transform(self):
        als = Als()
        output = als.fit(self.input_table).transform(self.test_table)[0]
        self.verify_output_result(
            output,
            als.get_prediction_col(),
            output.get_schema().get_field_names())

    def test_save_load_and_transform(self):
        als = Als()
        reloaded_Als = self.save_and_reload(Als)
        output = reloaded_Als.fit(self.input_table).transform(self.test_table)[0]
        self.verify_output_result(
            output,
            als.get_prediction_col(),
            output.get_schema().get_field_names())

    def verify_output_result(
            self, output: Table,
            prediction_col: str,
            field_names: List[str]):
        collected_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]
        for result in collected_results:
            prediction = result[field_names.index(prediction_col)]
            self.assertEqual(0.37558552399494904, prediction)
