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

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.idf import IDF, IDFModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params


class IDFTest(PyFlinkMLTestCase):
    def setUp(self):
        super(IDFTest, self).setUp()
        self.input_data = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(0, 1, 0, 2),),
                (Vectors.dense(0, 1, 2, 3),),
                (Vectors.dense(0, 1, 0, 0),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input', ],
                    [DenseVectorTypeInfo(), ])))

        self.expected_output = [
            Vectors.dense(0.0, 0.0, 0.0, 0.5753641),
            Vectors.dense(0.0, 0.0, 1.3862943, 0.8630462),
            Vectors.dense(0.0, 0.0, 0.0, 0.0),
        ]

        self.expected_output_min_doc_freq_as_two = [
            Vectors.dense(0.0, 0.0, 0.0, 0.5753641),
            Vectors.dense(0.0, 0.0, 0.0, 0.8630462),
            Vectors.dense(0.0, 0.0, 0.0, 0.0),
        ]

        self.tolerance = 1e-7

    def verify_prediction_result(self, expected, output_table):
        predicted_results = [result[1] for result in
                             self.t_env.to_data_stream(output_table).execute_and_collect()]

        predicted_results.sort(key=lambda x: x[3])
        expected.sort(key=lambda x: x[3])

        self.assertEqual(len(expected), len(predicted_results))
        for i in range(len(expected)):
            expected_row = expected[i]
            predicted_row = predicted_results[i]
            self.assertEqual(len(expected_row), len(predicted_row))
            for j in range(len(expected_row)):
                self.assertAlmostEqual(expected_row[j], predicted_row[j], delta=self.tolerance)

    def test_param(self):
        idf = IDF()
        self.assertEqual("input", idf.input_col)
        self.assertEqual(0, idf.min_doc_freq)
        self.assertEqual("output", idf.output_col)

        idf \
            .set_input_col("test_input") \
            .set_min_doc_freq(2) \
            .set_output_col("test_output")

        self.assertEqual("test_input", idf.input_col)
        self.assertEqual(2, idf.min_doc_freq)
        self.assertEqual("test_output", idf.output_col)

    def test_output_schema(self):
        idf = IDF() \
            .set_input_col("test_input") \
            .set_output_col("test_output")
        input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(1), ''),
            ],
                type_info=Types.ROW_NAMED(
                    ['test_input', 'dummy_input'],
                    [DenseVectorTypeInfo(), Types.STRING()])))
        output = idf \
            .fit(input_data_table) \
            .transform(input_data_table)[0]

        self.assertEqual(
            [idf.input_col, 'dummy_input', idf.output_col],
            output.get_schema().get_field_names())

    def test_fit_and_predict(self):
        idf = IDF()
        # Tests minDocFreq = 0.
        output = idf.fit(self.input_data).transform(self.input_data)[0]
        self.verify_prediction_result(self.expected_output, output)

        # Tests minDocFreq = 2.
        idf.set_min_doc_freq(2)
        output = idf.fit(self.input_data).transform(self.input_data)[0]
        self.verify_prediction_result(self.expected_output_min_doc_freq_as_two, output)

    def test_get_model_data(self):
        idf = IDF()
        model = idf.fit(self.input_data)
        model_data = model.get_model_data()[0]
        expected_field_names = ['idf', 'docFreq', 'numDocs']
        self.assertEqual(expected_field_names, model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(1, len(model_rows))
        self.assertEqual(3, model_rows[0][expected_field_names.index('numDocs')])
        self.assertListEqual([0, 3, 1, 2], model_rows[0][expected_field_names.index('docFreq')])
        self.assertListAlmostEqual(
            [1.3862943, 0, 0.6931471, 0.2876820],
            model_rows[0][expected_field_names.index('idf')].to_array(),
            delta=self.tolerance)

    def test_set_model_data(self):
        idf = IDF()
        model_a = idf.fit(self.input_data)
        model_data = model_a.get_model_data()[0]

        model_b = IDFModel().set_model_data(model_data)
        update_existing_params(model_b, model_a)

        output = model_b.transform(self.input_data)[0]
        self.verify_prediction_result(self.expected_output, output)

    def test_save_load_predict(self):
        idf = IDF()
        estimator_path = os.path.join(self.temp_dir, 'test_save_load_predict_idf')
        idf.save(estimator_path)
        idf = IDF.load(self.t_env, estimator_path)

        model = idf.fit(self.input_data)
        model_path = os.path.join(self.temp_dir, 'test_save_load_predict_idf_model')
        model.save(model_path)
        self.env.execute('save_model')
        model = IDFModel.load(self.t_env, model_path)
        output = model.transform(self.input_data)[0]

        self.verify_prediction_result(self.expected_output, output)
