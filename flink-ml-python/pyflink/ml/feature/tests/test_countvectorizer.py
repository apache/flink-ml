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
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase

from pyflink.ml.linalg import Vectors, DenseVector

from pyflink.ml.feature.countvectorizer import CountVectorizer, CountVectorizerModel
from pyflink.table import Table


class CountVectorizerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(CountVectorizerTest, self).setUp()
        self.input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (1, ['a', 'c', 'b', 'c'],),
                (2, ['c', 'd', 'e'],),
                (3, ['a', 'b', 'c'],),
                (4, ['e', 'f'],),
                (5, ['a', 'c', 'a'],),
            ],
                type_info=Types.ROW_NAMED(
                    ['id', 'input', ],
                    [Types.INT(), Types.OBJECT_ARRAY(Types.STRING())])))

        self.expected_output = [
            Vectors.sparse(6, [0, 1, 2], [2.0, 1.0, 1.0]),
            Vectors.sparse(6, [0, 3, 4], [1.0, 1.0, 1.0]),
            Vectors.sparse(6, [0, 1, 2], [1.0, 1.0, 1.0]),
            Vectors.sparse(6, [3, 5], [1.0, 1.0]),
            Vectors.sparse(6, [0, 1], [1.0, 2.0]),
        ]

    def test_param(self):
        count_vectorizer = CountVectorizer()
        self.assertEqual('input', count_vectorizer.input_col)
        self.assertEqual('output', count_vectorizer.output_col)
        self.assertEqual(1, count_vectorizer.min_df)
        self.assertEqual(float(2**63 - 1), count_vectorizer.max_df)
        self.assertEqual(1, count_vectorizer.min_tf)
        self.assertEqual(1 << 18, count_vectorizer.vocabulary_size)
        self.assertFalse(count_vectorizer.binary)

        count_vectorizer.\
            set_input_col('test_input').\
            set_output_col('test_output').\
            set_min_df(0.1).\
            set_max_df(0.9).\
            set_min_tf(10).\
            set_vocabulary_size(1000).\
            set_binary(True)
        self.assertEqual('test_input', count_vectorizer.input_col)
        self.assertEqual('test_output', count_vectorizer.output_col)
        self.assertEqual(0.1, count_vectorizer.min_df)
        self.assertEqual(0.9, count_vectorizer.max_df)
        self.assertEqual(10, count_vectorizer.min_tf)
        self.assertEqual(1000, count_vectorizer.vocabulary_size)
        self.assertTrue(count_vectorizer.binary)

    def test_output_schema(self):
        count_vectorizer = CountVectorizer()
        model = count_vectorizer.fit(self.input_table)
        output = model.transform(self.input_table.alias('id', 'input'))[0]
        self.assertEqual(
            ['id', 'input', 'output'],
            output.get_schema().get_field_names())

    def test_fit_and_predict(self):
        count_vectorizer = CountVectorizer()
        model = count_vectorizer.fit(self.input_table)
        output = model.transform(self.input_table)[0]
        self.verify_output_result(
            output,
            count_vectorizer.get_output_col(),
            output.get_schema().get_field_names(),
            self.expected_output)

    def test_save_load_predict(self):
        count_vectorizer = CountVectorizer()
        reloaded_count_vectorizer = self.save_and_reload(count_vectorizer)
        model = reloaded_count_vectorizer.fit(self.input_table)
        reloaded_model = self.save_and_reload(model)
        output = reloaded_model.transform(self.input_table)[0]
        self.verify_output_result(
            output,
            count_vectorizer.get_output_col(),
            output.get_schema().get_field_names(),
            self.expected_output)

    def test_get_model_data(self):
        count_vectorizer = CountVectorizer()
        model = count_vectorizer.fit(self.input_table)
        model_data_table = model.get_model_data()[0]
        self.assertEqual(["vocabulary"],
                         model_data_table.get_schema().get_field_names())
        model_data = self.t_env.to_data_stream(model_data_table).execute_and_collect().next()
        expected = ["c", "a", "b", "e", "d", "f"]
        self.assertEqual(expected, model_data[0])

    def test_set_model_data(self):
        count_vectorizer = CountVectorizer()
        model = count_vectorizer.fit(self.input_table)

        new_model = CountVectorizerModel()
        new_model.set_model_data(*model.get_model_data())
        output = new_model.transform(self.input_table)[0]
        self.verify_output_result(
            output,
            count_vectorizer.get_output_col(),
            output.get_schema().get_field_names(),
            self.expected_output)

    def verify_output_result(
            self, output: Table,
            output_col: str,
            field_names: List[str],
            expected_result: List[DenseVector]):
        collected_results = [result for result in
                             self.t_env.to_data_stream(output).execute_and_collect()]
        results = []
        for item in collected_results:
            item.set_field_names(field_names)
            results.append(item)
        results.sort(key=lambda x: x[0])
        results = list(map(lambda x: x[output_col], results))
        self.assertEqual(expected_result, results)
