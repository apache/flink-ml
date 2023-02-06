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
from typing import List

import numpy as np
from pyflink.table import Table
from pyflink.common import Types, Row
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params
from pyflink.ml.feature.imputer import Imputer, ImputerModel


class ImputerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(ImputerTest, self).setUp()
        self.train_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (float('NaN'), 9.0, 1,),
                (1.0, 9.0, None),
                (1.5, 7.0, 1,),
                (1.5, float('NaN'), 2,),
                (4.0, 5.0, 4,),
                (None, 4.0, None,),
            ],
                type_info=Types.ROW_NAMED(
                    ['f1', 'f2', 'f3'],
                    [Types.DOUBLE(), Types.DOUBLE(), Types.INT()])
            ))
        self.expected_mean_strategy_output = [
            Row(2.0, 9.0, 1.0,),
            Row(1.0, 9.0, 2.0,),
            Row(1.5, 7.0, 1.0,),
            Row(1.5, 6.8, 2.0,),
            Row(4.0, 5.0, 4.0,),
            Row(2.0, 4.0, 2.0,),
        ]
        self.expected_median_strategy_output = [
            Row(1.5, 9.0, 1.0,),
            Row(1.0, 9.0, 1.0,),
            Row(1.5, 7.0, 1.0,),
            Row(1.5, 7.0, 2.0,),
            Row(4.0, 5.0, 4.0,),
            Row(1.5, 4.0, 1.0,),
        ]
        self.expected_most_frequent_strategy_output = [
            Row(1.5, 9.0, 1.0,),
            Row(1.0, 9.0, 1.0,),
            Row(1.5, 7.0, 1.0,),
            Row(1.5, 9.0, 2.0,),
            Row(4.0, 5.0, 4.0,),
            Row(1.5, 4.0, 1.0,),
        ]
        self.strategy_and_expected_outputs = {
            'mean': self.expected_mean_strategy_output,
            'median': self.expected_median_strategy_output,
            'most_frequent': self.expected_most_frequent_strategy_output
        }
        self.eps = 1e-5

    def test_param(self):
        imputer = Imputer().\
            set_input_cols('f1', 'f2', 'f3').\
            set_output_cols('o1', 'o2', 'o3')

        self.assertEqual(('f1', 'f2', 'f3'), imputer.input_cols)
        self.assertEqual(('o1', 'o2', 'o3'), imputer.output_cols)
        self.assertEqual('mean', imputer.strategy)
        self.assertTrue(np.isnan(imputer.missing_value))

        imputer.set_strategy('median').set_missing_value(1.0)
        self.assertEqual('median', imputer.strategy)
        self.assertEqual(1.0, imputer.missing_value)

    def test_output_schema(self):
        imputer = Imputer().\
            set_input_cols('f1', 'f2', 'f3').\
            set_output_cols('o1', 'o2', 'o3')

        model = imputer.fit(self.train_table)
        output = model.transform(self.train_table)[0]
        self.assertEqual(
            ['f1', 'f2', 'f3', 'o1', 'o2', 'o3'],
            output.get_schema().get_field_names())

    def test_fit_and_predict(self):
        for strategy, expected_output in self.strategy_and_expected_outputs.items():
            imputer = Imputer().\
                set_input_cols('f1', 'f2', 'f3').\
                set_output_cols('o1', 'o2', 'o3').\
                set_strategy(strategy)
            model = imputer.fit(self.train_table)
            output = model.transform(self.train_table)[0]
            field_names = output.get_schema().get_field_names()
            self.verify_output_result(
                output, imputer.get_output_cols(), field_names, expected_output)

    def test_get_model_data(self):
        imputer = Imputer().\
            set_input_cols('f1', 'f2', 'f3').\
            set_output_cols('o1', 'o2', 'o3')
        model = imputer.fit(self.train_table)
        model_data = model.get_model_data()[0]
        expected_field_names = ['surrogates']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        surrogates = model_rows[0][expected_field_names.index('surrogates')]
        self.assertAlmostEqual(2.0, surrogates['f1'], delta=self.eps)
        self.assertAlmostEqual(6.8, surrogates['f2'], delta=self.eps)
        self.assertAlmostEqual(2.0, surrogates['f3'], delta=self.eps)

    def test_set_model_data(self):
        imputer = Imputer().\
            set_input_cols('f1', 'f2', 'f3').\
            set_output_cols('o1', 'o2', 'o3')
        model_a = imputer.fit(self.train_table)
        model_data = model_a.get_model_data()[0]

        model_b = ImputerModel().set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output = model_b.transform(self.train_table)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_output_result(
            output, imputer.get_output_cols(), field_names, self.expected_mean_strategy_output)

    def test_save_load_predict(self):
        imputer = Imputer(). \
            set_input_cols('f1', 'f2', 'f3'). \
            set_output_cols('o1', 'o2', 'o3')
        reloaded_imputer = self.save_and_reload(imputer)
        model = reloaded_imputer.fit(self.train_table)
        reloaded_model = self.save_and_reload(model)
        output = reloaded_model.transform(self.train_table)[0]
        self.verify_output_result(
            output,
            imputer.get_output_cols(),
            output.get_schema().get_field_names(),
            self.expected_mean_strategy_output)

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
            fields = []
            for col in output_cols:
                fields.append(item[col])
            results.append(Row(*fields))
        self.assertEqual(expected_result.sort(key=lambda x: str(x)),
                         results.sort(key=lambda x: str(x)))
