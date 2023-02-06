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

from pyflink.common import Row
from pyflink.table import DataTypes, Table

from pyflink.ml.feature.bucketizer import Bucketizer
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class BucketizerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(BucketizerTest, self).setUp()
        self.input_table = self.t_env.from_elements([
            (1, -0.5, 0.0, 1.0),
            (2, float('-inf'), 1.0, float('inf')),
            (3, float('nan'), -0.5, -0.5)],
            DataTypes.ROW(
                [DataTypes.FIELD("id", DataTypes.INT()),
                 DataTypes.FIELD("f1", DataTypes.DOUBLE()),
                 DataTypes.FIELD("f2", DataTypes.DOUBLE()),
                 DataTypes.FIELD("f3", DataTypes.DOUBLE())]))
        self.splits_array = ((-0.5, 0.0, 0.5),
                             (-1.0, 0.0, 2.0),
                             (float('-inf'), 10.0, float('inf')))
        self.expected_keep_result = [Row(1, 0, 1, 0), Row(2, 2, 1, 1), Row(3, 2, 0, 0)]
        self.expected_skip_result = [Row(1, 0, 1, 0)]

    def test_param(self):
        bucketizer = Bucketizer()
        self.assertEqual("error", bucketizer.handle_invalid)

        bucketizer.set_input_cols("f1", "f2", "f3") \
            .set_output_cols("o1", "o2", "o3") \
            .set_handle_invalid("skip") \
            .set_splits_array(self.splits_array)

        self.assertEqual(('f1', 'f2', 'f3'), bucketizer.input_cols)
        self.assertEqual(('o1', 'o2', 'o3'), bucketizer.output_cols)
        self.assertEqual('skip', bucketizer.handle_invalid)

        self.assertEqual(self.splits_array, bucketizer.get_split_array())

    def test_output_schema(self):
        bucketizer = Bucketizer()
        bucketizer.set_input_cols('f1', 'f2', 'f3') \
            .set_output_cols('o1', 'o2', 'o3') \
            .set_handle_invalid('skip') \
            .set_splits_array(self.splits_array)

        output = bucketizer.transform(self.input_table)[0]
        self.assertEqual(
            ['id', 'f1', 'f2', 'f3', 'o1', 'o2', 'o3'],
            output.get_schema().get_field_names())

    def test_transform(self):
        bucketizer = Bucketizer() \
            .set_input_cols('f1', 'f2', 'f3') \
            .set_output_cols('o1', 'o2', 'o3') \
            .set_splits_array(self.splits_array)

        # Tests skip.
        bucketizer.set_handle_invalid('skip')
        output = bucketizer.transform(self.input_table)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_output_result(
            output, bucketizer.get_output_cols(), field_names, self.expected_skip_result)

        # Tests keep
        bucketizer.set_handle_invalid('keep')
        output = bucketizer.transform(self.input_table)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_output_result(
            output, bucketizer.get_output_cols(), field_names, self.expected_keep_result)

    def test_save_load_and_transform(self):
        bucketizer = Bucketizer() \
            .set_input_cols('f1', 'f2', 'f3') \
            .set_output_cols('o1', 'o2', 'o3') \
            .set_handle_invalid('keep') \
            .set_splits_array(self.splits_array)

        path = os.path.join(self.temp_dir, 'test_save_load_and_transform_bucketizer')
        bucketizer.save(path)
        loaded_bucketizer = Bucketizer.load(self.t_env, path)
        output = loaded_bucketizer.transform(self.input_table)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_output_result(
            output, bucketizer.get_output_cols(), field_names, self.expected_keep_result)

    def verify_output_result(
            self, output: Table,
            output_cols: List[str],
            field_names: List[str],
            expected_result: List[Row]):
        collected_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]
        results = []
        for item in collected_results:
            item.set_field_names(field_names)
            fields = [item['id']]
            for col in output_cols:
                fields.append(item[col])
            results.append(Row(*fields))
        self.assertEqual(expected_result, results)
