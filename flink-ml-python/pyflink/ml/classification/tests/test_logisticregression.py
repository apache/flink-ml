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

from pyflink.common import Types
from pyflink.table import Table

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo, DenseVector
from pyflink.ml.classification.logisticregression import LogisticRegression, \
    LogisticRegressionModel, OnlineLogisticRegression
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class LogisticRegressionTest(PyFlinkMLTestCase):
    def setUp(self):
        super(LogisticRegressionTest, self).setUp()
        self.binomial_data_table = self.t_env.from_data_stream(
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
                    [DenseVectorTypeInfo(), Types.DOUBLE(), Types.DOUBLE()])
            ))

    def test_param(self):
        regression = LogisticRegression()
        self.assertEqual(regression.label_col, 'label')
        self.assertIsNone(regression.weight_col)
        self.assertEqual(regression.max_iter, 20)
        self.assertAlmostEqual(regression.reg, 0, delta=1e-7)
        self.assertAlmostEqual(regression.learning_rate, 0.1, delta=1e-7)
        self.assertEqual(regression.global_batch_size, 32)
        self.assertAlmostEqual(regression.tol, 1e-6, delta=1e-7)
        self.assertEqual(regression.multi_class, 'auto')
        self.assertEqual(regression.features_col, 'features')
        self.assertEqual(regression.prediction_col, 'prediction')
        self.assertEqual(regression.raw_prediction_col, 'rawPrediction')

        regression.set_features_col("test_features") \
            .set_label_col("test_label") \
            .set_weight_col("test_weight") \
            .set_max_iter(1000) \
            .set_tol(0.001) \
            .set_learning_rate(0.5) \
            .set_global_batch_size(1000) \
            .set_reg(0.1) \
            .set_multi_class("binomial") \
            .set_prediction_col("test_prediction_col") \
            .set_raw_prediction_col("test_raw_prediction_col")

        self.assertEqual(regression.features_col, 'test_features')
        self.assertEqual(regression.label_col, 'test_label')
        self.assertEqual(regression.weight_col, 'test_weight')
        self.assertEqual(regression.max_iter, 1000)
        self.assertAlmostEqual(regression.reg, 0.1, delta=1e-7)
        self.assertAlmostEqual(regression.learning_rate, 0.5, delta=1e-7)
        self.assertEqual(regression.global_batch_size, 1000)
        self.assertAlmostEqual(regression.tol, 0.001, delta=1e-7)
        self.assertEqual(regression.multi_class, 'binomial')
        self.assertEqual(regression.prediction_col, 'test_prediction_col')
        self.assertEqual(regression.raw_prediction_col, 'test_raw_prediction_col')

    def test_output_schema(self):
        temp_table = self.binomial_data_table.alias("test_features", "test_label", "test_weight")
        regression = LogisticRegression() \
            .set_features_col('test_features') \
            .set_label_col('test_label') \
            .set_weight_col('test_weight') \
            .set_prediction_col('test_prediction_col') \
            .set_raw_prediction_col('test_raw_prediction_col')
        output = regression.fit(self.binomial_data_table).transform(temp_table)[0]
        self.assertEqual(
            ['test_features',
             'test_label',
             'test_weight',
             'test_prediction_col',
             'test_raw_prediction_col'],
            output.get_schema().get_field_names())

    def test_fit_and_predict(self):
        regression = LogisticRegression().set_weight_col('weight')
        output = regression.fit(self.binomial_data_table).transform(self.binomial_data_table)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_predict_result(
            output,
            field_names.index(regression.get_features_col()),
            field_names.index(regression.get_prediction_col()),
            field_names.index(regression.get_raw_prediction_col()))

    def test_save_load_and_predict(self):
        regression = LogisticRegression().set_weight_col('weight')
        path = os.path.join(self.temp_dir, 'test_save_load_and_predict_logistic_regression')
        regression.save(path)
        regression = LogisticRegression.load(self.t_env, path)  # type: LogisticRegression
        model = regression.fit(self.binomial_data_table)
        self.assertEqual(
            model.get_model_data()[0].get_schema().get_field_names(),
            ['coefficient', 'modelVersion'])
        output = model.transform(self.binomial_data_table)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_predict_result(
            output,
            field_names.index(regression.get_features_col()),
            field_names.index(regression.get_prediction_col()),
            field_names.index(regression.get_raw_prediction_col()))

    def test_get_model_data(self):
        regression = LogisticRegression().set_weight_col('weight')
        model = regression.fit(self.binomial_data_table)
        model_data = self.t_env.to_data_stream(
            model.get_model_data()[0]).execute_and_collect().next()
        self.assertIsNotNone(model_data[0])
        data = model_data[0].values.tolist()
        expected = [0.528, -0.286, -0.429, -0.572]
        self.assertEqual(len(data), len(expected))
        for a, b in zip(data, expected):
            self.assertAlmostEqual(a, b, delta=0.1)

    def test_set_model_data(self):
        regression = LogisticRegression().set_weight_col('weight')
        model = regression.fit(self.binomial_data_table)

        new_model = LogisticRegressionModel()
        new_model.set_model_data(*model.get_model_data())
        output = new_model.transform(self.binomial_data_table)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_predict_result(
            output,
            field_names.index(regression.get_features_col()),
            field_names.index(regression.get_prediction_col()),
            field_names.index(regression.get_raw_prediction_col()))

    def verify_predict_result(
            self, output: Table, feature_index, prediction_index, raw_prediction_index):
        with self.t_env.to_data_stream(output).execute_and_collect() as results:
            for result in results:
                feature = result[feature_index]  # type: DenseVector
                prediction = result[prediction_index]  # type: float
                raw_prediction = result[raw_prediction_index]  # type: DenseVector
                if feature.get(0) <= 5:
                    self.assertAlmostEqual(0, prediction, delta=1e-7)
                    self.assertTrue(raw_prediction.get(0) > 0.5)
                else:
                    self.assertAlmostEqual(1, prediction, delta=1e-7)
                    self.assertTrue(raw_prediction.get(0) < 0.5)


class OnlineLogisticRegressionTest(PyFlinkMLTestCase):

    def setUp(self):
        super(OnlineLogisticRegressionTest, self).setUp()

    def test_param(self):
        online_logistic_regression = OnlineLogisticRegression()
        self.assertEqual("features", online_logistic_regression.features_col)
        self.assertEqual("count", online_logistic_regression.batch_strategy)
        self.assertEqual("label", online_logistic_regression.label_col)
        self.assertEqual(None, online_logistic_regression.weight_col)
        self.assertEqual(0.0, online_logistic_regression.reg)
        self.assertEqual(0.0, online_logistic_regression.elastic_net)
        self.assertEqual(0.1, online_logistic_regression.alpha)
        self.assertEqual(0.1, online_logistic_regression.beta)
        self.assertEqual(32, online_logistic_regression.global_batch_size)

        online_logistic_regression \
            .set_features_col("test_feature") \
            .set_label_col("test_label") \
            .set_global_batch_size(5) \
            .set_reg(0.5) \
            .set_elastic_net(0.25) \
            .set_alpha(0.1) \
            .set_beta(0.2)

        self.assertEqual("test_feature", online_logistic_regression.features_col)
        self.assertEqual("test_label", online_logistic_regression.label_col)
        self.assertEqual(0.5, online_logistic_regression.reg)
        self.assertEqual(0.25, online_logistic_regression.elastic_net)
        self.assertEqual(0.1, online_logistic_regression.alpha)
        self.assertEqual(0.2, online_logistic_regression.beta)
        self.assertEqual(5, online_logistic_regression.global_batch_size)
