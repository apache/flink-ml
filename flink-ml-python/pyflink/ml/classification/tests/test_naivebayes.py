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
from pyflink.ml.classification.naivebayes import NaiveBayes, NaiveBayesModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params


class NaiveBayesTest(PyFlinkMLTestCase):
    def setUp(self):
        super(NaiveBayesTest, self).setUp()
        self.env.set_parallelism(1)
        self.train_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([0, 0.]), 11.),
                (Vectors.dense([1, 0]), 10.),
                (Vectors.dense([1, 1.]), 10.),
            ],
                type_info=Types.ROW_NAMED(
                    ['features', 'label'],
                    [DenseVectorTypeInfo(), Types.DOUBLE()])))

        self.predict_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([0, 1.]),),
                (Vectors.dense([0, 0.]),),
                (Vectors.dense([1, 0]),),
                (Vectors.dense([1, 1.]),),
            ],
                type_info=Types.ROW_NAMED(
                    ['features'],
                    [DenseVectorTypeInfo()])))

        self.expected_output = {
            Vectors.dense([0, 1.]): 11.,
            Vectors.dense([0, 0.]): 11.,
            Vectors.dense([1, 0.]): 10.,
            Vectors.dense([1, 1.]): 10.,
        }

        self.estimator = NaiveBayes() \
            .set_smoothing(1.0) \
            .set_features_col('features') \
            .set_label_col('label') \
            .set_prediction_col('prediction') \
            .set_model_type('multinomial')  # type: NaiveBayes

    def test_param(self):
        estimator = NaiveBayes()

        self.assertEqual('features', estimator.get_features_col())
        self.assertEqual('label', estimator.get_label_col())
        self.assertEqual('multinomial', estimator.get_model_type())
        self.assertEqual('prediction', estimator.get_prediction_col())
        self.assertEqual(1.0, estimator.get_smoothing())

        estimator.set_features_col('test_feature') \
            .set_label_col('test_label') \
            .set_prediction_col('test_prediction') \
            .set_smoothing(2.0)

        self.assertEqual('test_feature', estimator.get_features_col())
        self.assertEqual('test_label', estimator.get_label_col())
        self.assertEqual('test_prediction', estimator.get_prediction_col())
        self.assertEqual(2.0, estimator.get_smoothing())

    def test_fit_and_predict(self):
        model = self.estimator.fit(self.train_data)  # type: NaiveBayesModel
        output_table = model.transform(self.predict_data)[0]
        actual_output = self.execute_and_collect(output_table)
        self.assertEqual(self.expected_output, actual_output)

    def test_output_schema(self):
        train_data = self.train_data.alias('test_features', 'test_label')
        predict_table = self.predict_data.alias('test_features')

        self.estimator \
            .set_features_col('test_features') \
            .set_label_col('test_label') \
            .set_prediction_col('test_prediction')

        model = self.estimator.fit(train_data)
        output_table = model.transform(predict_table)[0]
        actual_output = self.execute_and_collect(output_table)
        self.assertEqual(self.expected_output, actual_output)

    def test_save_load(self):
        path = os.path.join(self.temp_dir, 'test_save_load_naive_bayes')
        self.estimator.save(path)
        estimator = NaiveBayes.load(self.t_env, path)  # type: NaiveBayes
        model = estimator.fit(self.train_data)
        output_table = model.transform(self.predict_data)[0]
        actual_output = self.execute_and_collect(output_table)
        self.assertEqual(self.expected_output, actual_output)

    def test_get_model_data(self):
        train_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([1, 1.]), 11.),
                (Vectors.dense([2, 1]), 11.),
            ],
                type_info=Types.ROW_NAMED(
                    ['features', 'label'],
                    [DenseVectorTypeInfo(), Types.DOUBLE()])))

        model = self.estimator.fit(train_data)
        model_data = model.get_model_data()[0]
        expected_field_names = ['theta', 'piArray', 'labels']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        self.assertListAlmostEqual(
            [11.], model_rows[0][expected_field_names.index('labels')].to_array(), delta=1e-5)
        self.assertListAlmostEqual(
            [0.], model_rows[0][expected_field_names.index('piArray')].to_array(), delta=1e-5)
        theta = model_rows[0][expected_field_names.index('theta')]
        self.assertAlmostEqual(-0.6931471805599453, theta[0][0].get(1.0), delta=1e-5)
        self.assertAlmostEqual(-0.6931471805599453, theta[0][0].get(2.0), delta=1e-5)
        self.assertAlmostEqual(0.0, theta[0][1].get(1.0), delta=1e-5)

    def test_set_model_data(self):
        model_a = self.estimator.fit(self.train_data)
        model_data = model_a.get_model_data()[0]

        model_b = NaiveBayesModel().set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output_table = model_b.transform(self.predict_data)[0]
        actual_output = self.execute_and_collect(output_table)
        self.assertEqual(self.expected_output, actual_output)

    def execute_and_collect(self, output: Table):
        res = {}
        with self.t_env.to_data_stream(output).execute_and_collect() as results:
            for result in results:
                res[result[0]] = result[1]
        return res
