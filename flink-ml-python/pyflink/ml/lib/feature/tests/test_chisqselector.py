################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License") you may not use this file except in compliance
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
import sys

from pyflink.common import Types
from pyflink.ml.core.linalg import Vectors, VectorTypeInfo
from pyflink.ml.lib.feature.chisqselector import ChiSqSelector, ChiSqSelectorModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase
from pyflink.table import DataTypes, Table
from pyflink.table.expressions import col, null_of


class ChiSqSelectorTest(PyFlinkMLTestCase):
    def setUp(self):
        super(ChiSqSelectorTest, self).setUp()
        self.input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (0.0, Vectors.sparse(6, [0, 1, 3, 4], [6.0, 7.0, 7.0, 6.0]),),
                (0.0, Vectors.sparse(6, [1, 2, 4, 5], [9.0, 6.0, 5.0, 9.0]),),
                (0.0, Vectors.sparse(6, [1, 2, 4, 5], [9.0, 3.0, 5.0, 5.0]),),
                (1.0, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0),),
                (2.0, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0),),
                (2.0, Vectors.dense(8.0, 9.0, 6.0, 4.0, 0.0, 0.0),),
            ],
                type_info=Types.ROW_NAMED(
                    ['label', 'features'],
                    [Types.DOUBLE(), VectorTypeInfo()])))

        self.eps = 1e-7

    def test_param(self):
        selector = ChiSqSelector()

        self.assertEqual("numTopFeatures", selector.selector_type)
        self.assertEqual(50, selector.num_top_features)
        self.assertEqual(0.1, selector.percentile, sys.float_info.epsilon)
        self.assertEqual(0.05, selector.fpr, sys.float_info.epsilon)
        self.assertEqual(0.05, selector.fdr, sys.float_info.epsilon)
        self.assertEqual(0.05, selector.fwe, sys.float_info.epsilon)
        self.assertEqual("features", selector.features_col)
        self.assertEqual("label", selector.label_col)
        self.assertEqual("output", selector.output_col)

        selector.set_selector_type("percentile") \
            .set_num_top_features(10) \
            .set_percentile(0.5) \
            .set_fpr(0.1) \
            .set_fdr(0.1) \
            .set_fwe(0.1) \
            .set_features_col("test_features") \
            .set_label_col("test_label") \
            .set_output_col("test_output")

        self.assertEqual("percentile", selector.selector_type)
        self.assertEqual(10, selector.num_top_features)
        self.assertEqual(0.5, selector.percentile, sys.float_info.epsilon)
        self.assertEqual(0.1, selector.fpr, sys.float_info.epsilon)
        self.assertEqual(0.1, selector.fdr, sys.float_info.epsilon)
        self.assertEqual(0.1, selector.fwe, sys.float_info.epsilon)
        self.assertEqual("test_features", selector.features_col)
        self.assertEqual("test_label", selector.label_col)
        self.assertEqual("test_output", selector.output_col)

    def test_output_schema(self):
        selector = ChiSqSelector() \
            .set_features_col('test_features') \
            .set_label_col('test_label') \
            .set_output_col('test_output')
        input_table = self.input_table.select(
            col('features').alias('test_features'),
            col('label').alias('test_label'),
            null_of(DataTypes.INT()).alias('dummy_input'))
        output_table = selector.fit(input_table).transform(input_table)[0]

        self.assertEqual(
            ['test_features', 'test_label', 'dummy_input', 'test_output'],
            output_table.get_schema().get_field_names())

    def test_fit_and_predict(self):
        selector = ChiSqSelector() \
            .set_features_col('features') \
            .set_label_col('label') \
            .set_output_col('prediction') \
            .set_selector_type('numTopFeatures') \
            .set_num_top_features(1)

        output_table = selector.fit(self.input_table).transform(self.input_table)[0]

        self.verify_prediction_result(output_table, 0)

    def test_save_load_predict(self):
        selector = ChiSqSelector() \
            .set_features_col('features') \
            .set_label_col('label') \
            .set_output_col('prediction') \
            .set_selector_type('numTopFeatures') \
            .set_num_top_features(1)

        estimator_path = os.path.join(self.temp_dir, 'test_save_load_predict_chisqselector')
        selector.save(estimator_path)
        selector = ChiSqSelector.load(self.t_env, estimator_path)

        model = selector.fit(self.input_table)
        model_path = os.path.join(self.temp_dir, 'test_save_load_predict_chisqselector_model')
        model.save(model_path)
        self.env.execute('save_model')
        model = ChiSqSelectorModel.load(self.t_env, model_path)
        output_table = model.transform(self.input_table)[0]

        self.verify_prediction_result(output_table, 0)

    def verify_prediction_result(self, output_table: Table, *args):
        results = [(result[1], result[2]) for result in
                   self.t_env.to_data_stream(output_table).execute_and_collect()]

        selected_indices = list(args)

        for result in results:
            input_data = result[0]
            output_data = result[1]
            self.assertEqual(len(selected_indices), len(output_data))

            for i in range(len(selected_indices)):
                self.assertAlmostEqual(
                    input_data.get(selected_indices[i]), output_data.get(i), self.eps)
