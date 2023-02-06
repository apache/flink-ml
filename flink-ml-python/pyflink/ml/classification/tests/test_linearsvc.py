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

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.classification.linearsvc import LinearSVC, LinearSVCModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params


class LinearSVCTest(PyFlinkMLTestCase):
    def setUp(self):
        super(LinearSVCTest, self).setUp()
        self.train_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([1, 2, 3, 4]), 0., 1.),
                (Vectors.dense([2, 2, 3, 4]), 0., 2.),
                (Vectors.dense([3, 2, 3, 4]), 0., 3.),
                (Vectors.dense([4, 2, 3, 4]), 0., 4.),
                (Vectors.dense([5, 2, 3, 4]), 0., 5.),
                (Vectors.dense([11, 2, 3, 4]), 1., 1.),
                (Vectors.dense([12, 2, 3, 4]), 1., 2.),
                (Vectors.dense([13, 2, 3, 4]), 1., 3.),
                (Vectors.dense([14, 2, 3, 4]), 1., 4.),
                (Vectors.dense([15, 2, 3, 4]), 1., 5.),
            ],
                type_info=Types.ROW_NAMED(
                    ['features', 'label', 'weight'],
                    [DenseVectorTypeInfo(), Types.DOUBLE(), Types.DOUBLE()])))

        self.eps = 0.1

    def test_param(self):
        linear_svc = LinearSVC()
        self.assertEqual('features', linear_svc.features_col)
        self.assertEqual('label', linear_svc.label_col)
        self.assertIsNone(linear_svc.weight_col)
        self.assertEqual(20, linear_svc.max_iter)
        self.assertAlmostEqual(1e-6, linear_svc.tol, delta=1e-7)
        self.assertAlmostEqual(0.1, linear_svc.learning_rate, delta=1e-7)
        self.assertEqual(32, linear_svc.global_batch_size)
        self.assertAlmostEqual(0, linear_svc.reg, delta=1e-7)
        self.assertAlmostEqual(0, linear_svc.elastic_net, delta=1e-7)
        self.assertAlmostEqual(0, linear_svc.threshold, delta=1e-7)
        self.assertEqual('prediction', linear_svc.prediction_col)
        self.assertEqual('rawPrediction', linear_svc.raw_prediction_col)

        linear_svc.set_features_col('test_features') \
            .set_label_col('test_label') \
            .set_weight_col('test_weight') \
            .set_max_iter(1000) \
            .set_tol(0.001) \
            .set_learning_rate(0.5) \
            .set_global_batch_size(1000) \
            .set_reg(0.1) \
            .set_elastic_net(0.5) \
            .set_threshold(0.5) \
            .set_prediction_col('test_predictionCol') \
            .set_raw_prediction_col('test_rawPredictionCol')

        self.assertEqual('test_features', linear_svc.features_col)
        self.assertEqual('test_label', linear_svc.label_col)
        self.assertEqual('test_weight', linear_svc.weight_col)
        self.assertEqual(1000, linear_svc.max_iter)
        self.assertAlmostEqual(0.001, linear_svc.tol, delta=1e-7)
        self.assertAlmostEqual(0.5, linear_svc.learning_rate, delta=1e-7)
        self.assertEqual(1000, linear_svc.global_batch_size)
        self.assertAlmostEqual(0.1, linear_svc.reg, delta=1e-7)
        self.assertAlmostEqual(0.5, linear_svc.elastic_net, delta=1e-7)
        self.assertAlmostEqual(0.5, linear_svc.threshold, delta=1e-7)
        self.assertEqual('test_predictionCol', linear_svc.prediction_col)
        self.assertEqual('test_rawPredictionCol', linear_svc.raw_prediction_col)

    def test_output_schema(self):
        temp_table = self.train_data.alias('test_features', 'test_label', 'test_weight')

        linear_svc = LinearSVC().set_features_col('test_features') \
            .set_label_col('test_label') \
            .set_weight_col('test_weight') \
            .set_prediction_col('test_predictionCol') \
            .set_raw_prediction_col('test_rawPredictionCol')

        output = linear_svc.fit(self.train_data).transform(temp_table)[0]
        self.assertEqual(
            ['test_features',
             'test_label',
             'test_weight',
             'test_predictionCol',
             'test_rawPredictionCol'],
            output.get_schema().get_field_names())

    def test_fit_and_predict(self):
        linear_svc = LinearSVC().set_weight_col('weight')
        output = linear_svc.fit(self.train_data).transform(self.train_data)[0]
        self.verify_prediction_result(
            output,
            output.get_schema().get_field_names(),
            linear_svc.features_col,
            linear_svc.prediction_col,
            linear_svc.raw_prediction_col)

    def test_get_model_data(self):
        linear_svc = LinearSVC().set_weight_col('weight')
        model_data = linear_svc.fit(self.train_data).get_model_data()[0]
        expected_field_names = ['coefficient']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        self.assertListAlmostEqual(
            [0.470, -0.273, -0.410, -0.546],
            model_rows[expected_field_names.index('coefficient')][0].to_array(),
            delta=self.eps)

    def test_set_model_data(self):
        linear_svc = LinearSVC().set_weight_col('weight')
        model_a = linear_svc.fit(self.train_data)
        model_data = model_a.get_model_data()[0]

        model_b = LinearSVCModel().set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output = model_b.transform(self.train_data)[0]
        self.verify_prediction_result(
            output,
            output.get_schema().get_field_names(),
            linear_svc.features_col,
            linear_svc.prediction_col,
            linear_svc.raw_prediction_col)

    def test_save_load_and_predict(self):
        linear_svc = LinearSVC().set_weight_col('weight')
        reloaded_linear_svc = self.save_and_reload(linear_svc)
        model = reloaded_linear_svc.fit(self.train_data)
        reloaded_model = self.save_and_reload(model)
        output = reloaded_model.transform(self.train_data)[0]
        self.verify_prediction_result(
            output,
            output.get_schema().get_field_names(),
            linear_svc.features_col,
            linear_svc.prediction_col,
            linear_svc.raw_prediction_col)

    def verify_prediction_result(self,
                                 output: Table,
                                 field_names,
                                 features_col,
                                 prediction_col,
                                 raw_prediction_col):
        collected_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]
        for item in collected_results:
            item.set_field_names(field_names)
            feature = item[features_col]
            prediction = item[prediction_col]
            raw_prediction = item[raw_prediction_col]
            if feature[0] <= 5:
                self.assertAlmostEqual(0, prediction, delta=1e-7)
                self.assertTrue(raw_prediction[0] < 0)
            else:
                self.assertAlmostEqual(1, prediction, delta=1e-7)
                self.assertTrue(raw_prediction[0] > 0)
