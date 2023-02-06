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

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo, DenseMatrix, DenseVector
from pyflink.ml.classification.knn import KNN, KNNModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class KNNTest(PyFlinkMLTestCase):
    def setUp(self):
        super(KNNTest, self).setUp()
        self.train_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([2.0, 3.0]), 1.0),
                (Vectors.dense([2.1, 3.1]), 1.0),
                (Vectors.dense([200.1, 300.1]), 2.0),
                (Vectors.dense([200.2, 300.2]), 2.0),
                (Vectors.dense([200.3, 300.3]), 2.0),
                (Vectors.dense([200.4, 300.4]), 2.0),
                (Vectors.dense([200.4, 300.4]), 2.0),
                (Vectors.dense([200.6, 300.6]), 2.0),
                (Vectors.dense([2.1, 3.1]), 1.0),
                (Vectors.dense([2.1, 3.1]), 1.0),
                (Vectors.dense([2.1, 3.1]), 1.0),
                (Vectors.dense([2.1, 3.1]), 1.0),
                (Vectors.dense([2.3, 3.2]), 1.0),
                (Vectors.dense([2.3, 3.2]), 1.0),
                (Vectors.dense([2.8, 3.2]), 3.0),
                (Vectors.dense([300., 3.2]), 4.0),
                (Vectors.dense([2.2, 3.2]), 1.0),
                (Vectors.dense([2.4, 3.2]), 5.0),
                (Vectors.dense([2.5, 3.2]), 5.0),
                (Vectors.dense([2.5, 3.2]), 5.0),
                (Vectors.dense([2.1, 3.1]), 1.0)
            ],
                type_info=Types.ROW_NAMED(
                    ['features', 'label'],
                    [DenseVectorTypeInfo(), Types.DOUBLE()])))

        self.predict_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense([4.0, 4.1]), 5.0),
                (Vectors.dense([300, 42]), 2.0),
            ],
                type_info=Types.ROW_NAMED(
                    ['features', 'label'],
                    [DenseVectorTypeInfo(), Types.DOUBLE()])))

    def test_param(self):
        knn = KNN()
        self.assertEqual('features', knn.get_features_col())
        self.assertEqual('label', knn.get_label_col())
        self.assertEqual(5, knn.get_k())
        self.assertEqual('prediction', knn.get_prediction_col())

        knn.set_label_col('test_label') \
            .set_features_col('test_features') \
            .set_k(4) \
            .set_prediction_col('test_prediction')

        self.assertEqual('test_features', knn.get_features_col())
        self.assertEqual('test_label', knn.get_label_col())
        self.assertEqual(4, knn.get_k())
        self.assertEqual('test_prediction', knn.get_prediction_col())

    def test_output_schema(self):
        knn = KNN() \
            .set_label_col('test_label') \
            .set_features_col('test_features') \
            .set_k(4) \
            .set_prediction_col('test_prediction')

        model = knn.fit(self.train_data.alias('test_features, test_label'))
        output = model.transform(self.predict_data.alias('test_features, test_label'))[0]
        self.assertEqual(output.get_schema().get_field_names(),
                         ['test_features',
                          'test_label',
                          'test_prediction'])

    def test_fewer_distinct_points_than_cluster(self):
        knn = KNN()
        model = knn.fit(self.predict_data)
        output = model.transform(self.predict_data)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_predict_result(
            output,
            field_names.index(knn.label_col),
            field_names.index(knn.prediction_col))

    def test_fit_and_predict(self):
        knn = KNN()
        model = knn.fit(self.train_data)
        output = model.transform(self.predict_data)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_predict_result(
            output,
            field_names.index(knn.label_col),
            field_names.index(knn.prediction_col))

    def test_save_load_and_predict(self):
        knn = KNN()
        path = os.path.join(self.temp_dir, 'test_save_load_and_predict_knn')
        knn.save(path)
        knn = KNN.load(self.t_env, path)  # type: KNN
        model = knn.fit(self.train_data)
        self.assertEqual(
            ["packedFeatures", "featureNormSquares", "labels"],
            model.get_model_data()[0].get_schema().get_field_names())

    def test_model_save_and_predict(self):
        knn = KNN()
        model = knn.fit(self.train_data)
        path = os.path.join(self.temp_dir, 'test_save_load_and_predict_knn_model')
        model.save(path)
        self.env.execute('test')
        new_model = model.load(self.t_env, path)
        output = new_model.transform(self.predict_data)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_predict_result(
            output,
            field_names.index(knn.label_col),
            field_names.index(knn.prediction_col))

    def test_get_model_data(self):
        knn = KNN()
        model = knn.fit(self.train_data)
        model_data = model.get_model_data()[0]
        output = self.t_env.to_data_stream(model_data)
        self.assertEqual('packedFeatures', model_data.get_schema().get_field_name(0))
        self.assertEqual('featureNormSquares', model_data.get_schema().get_field_name(1))
        self.assertEqual('labels', model_data.get_schema().get_field_name(2))
        with output.execute_and_collect() as results:
            model_rows = [result for result in results]
        packed_features = model_rows[0][0]  # type: DenseMatrix
        feature_norm_squares = model_rows[0][1]  # type: DenseVector
        labels = model_rows[0][2]  # type: DenseVector
        self.assertEqual(2, packed_features.num_rows())
        self.assertEqual(packed_features.num_cols(), labels.size())
        self.assertEqual(feature_norm_squares.size(), labels.size())

    def test_set_model_data(self):
        knn = KNN()
        model_a = knn.fit(self.train_data)
        model_data = model_a.get_model_data()[0]

        model_b = KNNModel().set_model_data(model_data)
        output = model_b.transform(self.predict_data)[0]
        field_names = output.get_schema().get_field_names()
        self.verify_predict_result(
            output,
            field_names.index(knn.label_col),
            field_names.index(knn.prediction_col))

    def verify_predict_result(
            self, output: Table, label_index, prediction_index):
        with self.t_env.to_data_stream(output).execute_and_collect() as results:
            for result in results:
                label = result[label_index]  # type: DenseVector
                prediction = result[prediction_index]  # type: float
                self.assertEqual(label, prediction)
