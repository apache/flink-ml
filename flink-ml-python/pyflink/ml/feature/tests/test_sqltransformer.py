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

from pyflink.common import Types, Row

from pyflink.ml.feature.sqltransformer import SQLTransformer
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class SQLTransformerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(SQLTransformerTest, self).setUp()
        self.input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (0, 1.0, 3.0),
                (2, 2.0, 5.0),
            ],
                type_info=Types.ROW_NAMED(
                    ['id', 'v1', 'v2'],
                    [Types.INT(), Types.DOUBLE(), Types.DOUBLE()])))
        self.expected_output = [
            (0, 1.0, 3.0, 4.0, 3.0),
            (2, 2.0, 5.0, 7.0, 10.0)
        ]

    def test_param(self):
        sql_transformer = SQLTransformer()
        sql_transformer.set_statement('SELECT * FROM __THIS__')
        self.assertEqual('SELECT * FROM __THIS__', sql_transformer.statement)

    def test_output_schema(self):
        sql_transformer = SQLTransformer() \
            .set_statement('SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__')
        output_table = sql_transformer.transform(self.input_table)[0]

        self.assertEqual(
            ['id', 'v1', 'v2', 'v3', 'v4'],
            output_table.get_schema().get_field_names())

    def test_save_load_transform(self):
        sql_transformer = SQLTransformer() \
            .set_statement('SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__')
        loaded_sql_transformer = self.save_and_reload(sql_transformer)
        output_table = loaded_sql_transformer.transform(self.input_table)[0]
        actual_output = [output for output in
                         self.t_env.to_data_stream(output_table).execute_and_collect()]
        actual_output.sort(key=lambda x: x[0])
        self.assertEqual(len(self.expected_output), len(actual_output))
        for i in range(len(actual_output)):
            self.assertEqual(Row(*self.expected_output[i]), actual_output[i])
