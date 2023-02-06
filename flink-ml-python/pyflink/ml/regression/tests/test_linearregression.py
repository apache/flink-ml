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

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.regression.linearregression import LinearRegression, \
    LinearRegressionModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class LinearRegressionTest(PyFlinkMLTestCase):
    def setUp(self):
        super(LinearRegressionTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(2, 1), 4., 1.),
                (Vectors.dense(3, 2), 7., 1.),
                (Vectors.dense(4, 3), 10., 1.),
                (Vectors.dense(2, 4), 10., 1.),
                (Vectors.dense(2, 2), 6., 1.),
                (Vectors.dense(4, 3), 10., 1.),
                (Vectors.dense(1, 2), 5., 1.),
                (Vectors.dense(5, 3), 11., 1.),
            ],
                type_info=Types.ROW_NAMED(
                    ['features', 'label', 'weight'],
                    [DenseVectorTypeInfo(), Types.DOUBLE(), Types.DOUBLE()])
            ))

    def test_param(self):
        regression = LinearRegression()
        self.assertEqual(regression.label_col, 'label')
        self.assertIsNone(regression.weight_col)
        self.assertEqual(regression.max_iter, 20)
        self.assertAlmostEqual(regression.reg, 0, delta=1e-7)
        self.assertAlmostEqual(regression.learning_rate, 0.1, delta=1e-7)
        self.assertEqual(regression.global_batch_size, 32)
        self.assertAlmostEqual(regression.tol, 1e-6, delta=1e-7)
        self.assertEqual(regression.features_col, 'features')
        self.assertEqual(regression.prediction_col, 'prediction')

        regression.set_features_col("test_features") \
            .set_label_col("test_label") \
            .set_weight_col("test_weight") \
            .set_max_iter(1000) \
            .set_tol(0.001) \
            .set_learning_rate(0.5) \
            .set_global_batch_size(1000) \
            .set_reg(0.1) \
            .set_prediction_col("test_prediction_col") \

        self.assertEqual(regression.features_col, 'test_features')
        self.assertEqual(regression.label_col, 'test_label')
        self.assertEqual(regression.weight_col, 'test_weight')
        self.assertEqual(regression.max_iter, 1000)
        self.assertAlmostEqual(regression.reg, 0.1, delta=1e-7)
        self.assertAlmostEqual(regression.learning_rate, 0.5, delta=1e-7)
        self.assertEqual(regression.global_batch_size, 1000)
        self.assertAlmostEqual(regression.tol, 0.001, delta=1e-7)
        self.assertEqual(regression.prediction_col, 'test_prediction_col')

    def test_output_schema(self):
        temp_table = self.input_data_table.alias("test_features", "test_label", "test_weight")
        regression = LinearRegression() \
            .set_features_col('test_features') \
            .set_label_col('test_label') \
            .set_weight_col('test_weight') \
            .set_prediction_col('test_prediction_col')
        output = regression.fit(self.input_data_table).transform(temp_table)[0]
        self.assertEqual(
            ['test_features',
             'test_label',
             'test_weight',
             'test_prediction_col'],
            output.get_schema().get_field_names())

    def test_fit_and_predict(self):
        regression = LinearRegression().set_weight_col('weight')
        output = regression.fit(self.input_data_table).transform(self.input_data_table)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_predict_result(
            output,
            field_names.index(regression.get_label_col()),
            field_names.index(regression.get_prediction_col()))

    def test_save_load_and_predict(self):
        regression = LinearRegression().set_weight_col('weight')
        path = os.path.join(self.temp_dir, 'test_save_load_and_predict_linear_regression')
        regression.save(path)
        regression = LinearRegression.load(self.t_env, path)  # type: LinearRegression
        model = regression.fit(self.input_data_table)
        self.assertEqual(
            model.get_model_data()[0].get_schema().get_field_names(),
            ['coefficient'])
        output = model.transform(self.input_data_table)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_predict_result(
            output,
            field_names.index(regression.get_label_col()),
            field_names.index(regression.get_prediction_col()))

    def test_get_model_data(self):
        regression = LinearRegression().set_weight_col('weight')
        model = regression.fit(self.input_data_table)
        model_data = self.t_env.to_data_stream(
            model.get_model_data()[0]).execute_and_collect().next()
        self.assertIsNotNone(model_data[0])
        data = model_data[0].values.tolist()
        expected = [1.141, 1.829]
        self.assertEqual(len(data), len(expected))
        for a, b in zip(data, expected):
            self.assertAlmostEqual(a, b, delta=0.1)

    def test_set_model_data(self):
        regression = LinearRegression().set_weight_col('weight')
        model = regression.fit(self.input_data_table)

        new_model = LinearRegressionModel()
        new_model.set_model_data(*model.get_model_data())
        output = new_model.transform(self.input_data_table)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_predict_result(
            output,
            field_names.index(regression.get_label_col()),
            field_names.index(regression.get_prediction_col()))

    def verify_predict_result(
            self, output: Table, label_index, prediction_index):
        with self.t_env.to_data_stream(output).execute_and_collect() as results:
            for result in results:
                label = result[label_index]  # type: float
                prediction = result[prediction_index]  # type: float
                self.assertAlmostEqual(label, prediction, delta=0.1 * label)
