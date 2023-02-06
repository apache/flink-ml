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

from pyflink.common import Types
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params

from pyflink.ml.linalg import DenseVectorTypeInfo, Vectors

from pyflink.ml.feature.univariatefeatureselector import UnivariateFeatureSelector, \
    UnivariateFeatureSelectorModel
from pyflink.table import Table


class UnivariateFeatureSelectorTest(PyFlinkMLTestCase):

    def setUp(self):
        super(UnivariateFeatureSelectorTest, self).setUp()
        self.input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (1, Vectors.dense(4.65415496e-03, 1.03550567e-01, -1.17358140e+00,
                                  1.61408773e-01, 3.92492111e-01, 7.31240882e-01)),
                (1, Vectors.dense(-9.01651741e-01, -5.28905302e-01, 1.27636785e+00,
                                  7.02154563e-01, 6.21348351e-01, 1.88397353e-01)),
                (1, Vectors.dense(3.85692159e-01, -9.04639637e-01, 5.09782604e-02,
                                  8.40043971e-01, 7.45977857e-01, 8.78402288e-01)),
                (1, Vectors.dense(1.36264353e+00, 2.62454094e-01, 7.96306202e-01,
                                  6.14948000e-01, 7.44948187e-01, 9.74034830e-01)),
                (1, Vectors.dense(9.65874070e-01, 2.52773665e+00, -2.19380094e+00,
                                  2.33408080e-01, 1.86340919e-01, 8.23390433e-01)),
                (2, Vectors.dense(1.12324305e+01, -2.77121515e-01, 1.12740513e-01,
                                  2.35184013e-01, 3.46668895e-01, 9.38500782e-02)),
                (2, Vectors.dense(1.06195839e+01, -1.82891238e+00, 2.25085601e-01,
                                  9.09979851e-01, 6.80257535e-02, 8.24017480e-01)),
                (2, Vectors.dense(1.12806837e+01, 1.30686889e+00, 9.32839108e-02,
                                  3.49784755e-01, 1.71322408e-02, 7.48465194e-02)),
                (2, Vectors.dense(9.98689462e+00, 9.50808938e-01, -2.90786359e-01,
                                  2.31253009e-01, 7.46270968e-01, 1.60308169e-01)),
                (2, Vectors.dense(1.08428551e+01, -1.02749936e+00, 1.73951508e-01,
                                  8.92482744e-02, 1.42651730e-01, 7.66751625e-01)),
                (3, Vectors.dense(-1.98641448e+00, 1.12811990e+01, -2.35246756e-01,
                                  8.22809049e-01, 3.26739456e-01, 7.88268404e-01)),
                (3, Vectors.dense(-6.09864090e-01, 1.07346276e+01, -2.18805509e-01,
                                  7.33931213e-01, 1.42554396e-01, 7.11225605e-01)),
                (3, Vectors.dense(-1.58481268e+00, 9.19364039e+00, -5.87490459e-02,
                                  2.51532056e-01, 2.82729807e-01, 7.16245686e-01)),
                (3, Vectors.dense(-2.50949277e-01, 1.12815254e+01, -6.94806734e-01,
                                  5.93898886e-01, 5.68425656e-01, 8.49762330e-01)),
                (3, Vectors.dense(7.63485129e-01, 1.02605138e+01, 1.32617719e+00,
                                  5.49682879e-01, 8.59931442e-01, 4.88677978e-02)),
                (4, Vectors.dense(9.34900015e-01, 4.11379043e-01, 8.65010205e+00,
                                  9.23509168e-01, 1.16995043e-01, 5.91894106e-03)),
                (4, Vectors.dense(4.73734933e-01, -1.48321181e+00, 9.73349621e+00,
                                  4.09421563e-01, 5.09375719e-01, 5.93157850e-01)),
                (4, Vectors.dense(3.41470679e-01, -6.88972582e-01, 9.60347938e+00,
                                  3.62654055e-01, 2.43437468e-01, 7.13052838e-01)),
                (4, Vectors.dense(-5.29614251e-01, -1.39262856e+00, 1.01354144e+01,
                                  8.24123861e-01, 5.84074506e-01, 6.54461558e-01)),
                (4, Vectors.dense(-2.99454508e-01, 2.20457263e+00, 1.14586015e+01,
                                  5.16336729e-01, 9.99776159e-01, 3.15769738e-01)),
            ],
                type_info=Types.ROW_NAMED(
                    ['label', 'features'],
                    [Types.INT(), DenseVectorTypeInfo()])
            ))

    def test_param(self):
        univariate_feature_selector = UnivariateFeatureSelector()
        self.assertEqual('features', univariate_feature_selector.features_col)
        self.assertEqual('label', univariate_feature_selector.label_col)
        self.assertEqual('output', univariate_feature_selector.output_col)
        with self.assertRaises(Exception) as context:
            univariate_feature_selector.feature_type
            self.assertTrue("Parameter featureType's value should not be null" in context.exception)
        with self.assertRaises(Exception) as context:
            univariate_feature_selector.label_type
            self.assertTrue("Parameter labelType's value should not be null" in context.exception)
        self.assertEqual('numTopFeatures', univariate_feature_selector.selection_mode)
        self.assertIsNone(univariate_feature_selector.selection_threshold)

        univariate_feature_selector\
            .set_features_col("test_features")\
            .set_label_col('test_label')\
            .set_output_col('test_output')\
            .set_feature_type('continuous')\
            .set_label_type('categorical')\
            .set_selection_mode('fpr')\
            .set_selection_threshold(0.01)
        self.assertEqual('test_features', univariate_feature_selector.features_col)
        self.assertEqual('test_label', univariate_feature_selector.label_col)
        self.assertEqual('test_output', univariate_feature_selector.output_col)
        self.assertEqual('continuous', univariate_feature_selector.feature_type)
        self.assertEqual('categorical', univariate_feature_selector.label_type)
        self.assertEqual('fpr', univariate_feature_selector.selection_mode)
        self.assertEqual(0.01, univariate_feature_selector.selection_threshold)

    def test_output_schema(self):
        selector = UnivariateFeatureSelector()\
            .set_features_col("test_features")\
            .set_label_col('test_label')\
            .set_output_col('test_output')\
            .set_feature_type('continuous')\
            .set_label_type('categorical')
        temp_table = self.input_table.alias('test_label', 'test_features')
        model = selector.fit(temp_table)
        output = model.transform(temp_table)[0]
        self.assertEqual(
            ['test_label', 'test_features', 'test_output'],
            output.get_schema().get_field_names())

    def test_fit_and_predict(self):
        selector = UnivariateFeatureSelector() \
            .set_feature_type('continuous') \
            .set_label_type('categorical') \
            .set_selection_threshold(3)
        model = selector.fit(self.input_table)
        output = model.transform(self.input_table)[0]
        self.verify_output_result(
            output,
            output.get_schema().get_field_names(),
            selector.get_features_col(),
            selector.get_output_col(),
            [0, 1, 2])

    def test_get_model_data(self):
        selector = UnivariateFeatureSelector() \
            .set_feature_type('continuous') \
            .set_label_type('categorical') \
            .set_selection_threshold(3)
        model = selector.fit(self.input_table)
        model_data = model.get_model_data()[0]
        self.assertEqual(['indices'], model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        self.assertListEqual([0, 2, 1], model_rows[0][0])

    def test_set_model_data(self):
        selector = UnivariateFeatureSelector() \
            .set_feature_type('continuous') \
            .set_label_type('categorical') \
            .set_selection_threshold(3)
        model_a = selector.fit(self.input_table)
        model_data = model_a.get_model_data()[0]

        model_b = UnivariateFeatureSelectorModel() \
            .set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output = model_b.transform(self.input_table)[0]
        self.verify_output_result(
            output,
            output.get_schema().get_field_names(),
            selector.get_features_col(),
            selector.get_output_col(),
            [0, 1, 2])

    def test_save_load_predict(self):
        selector = UnivariateFeatureSelector() \
            .set_feature_type('continuous') \
            .set_label_type('categorical') \
            .set_selection_threshold(3)
        reloaded_selector = self.save_and_reload(selector)
        model = reloaded_selector.fit(self.input_table)
        reloaded_model = self.save_and_reload(model)
        output = reloaded_model.transform(self.input_table)[0]
        self.verify_output_result(
            output,
            output.get_schema().get_field_names(),
            selector.get_features_col(),
            selector.get_output_col(),
            [0, 1, 2])

    def verify_output_result(
            self, output: Table,
            field_names: List[str],
            feature_col: str,
            output_col: str,
            indices: List[int]):
        collected_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]
        for item in collected_results:
            item.set_field_names(field_names)
            self.assertEqual(len(indices), item[output_col].size())
            for i in range(0, len(indices)):
                self.assertEqual(item[feature_col].get(indices[i]),
                                 item[output_col].get(i))
