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

from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.feature.vectorindexer import VectorIndexer, VectorIndexerModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class VectorIndexerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(VectorIndexerTest, self).setUp()
        self.train_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(1, 1),),
                (Vectors.dense(2, -1),),
                (Vectors.dense(3, 1),),
                (Vectors.dense(4, 0),),
                (Vectors.dense(5, 0),)
            ],
                type_info=Types.ROW_NAMED(
                    ['input', ],
                    [DenseVectorTypeInfo(), ])))

        self.predict_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(0, 2),),
                (Vectors.dense(0, 0),),
                (Vectors.dense(0, -1),),
            ],
                type_info=Types.ROW_NAMED(
                    ['input', ],
                    [DenseVectorTypeInfo(), ])))

        self.expected_output = [
            Vectors.dense(5, 3),
            Vectors.dense(5, 0),
            Vectors.dense(5, 1)]

    def test_param(self):
        vector_indexer = VectorIndexer()

        self.assertEqual('input', vector_indexer.input_col)
        self.assertEqual('output', vector_indexer.output_col)
        self.assertEqual(20, vector_indexer.max_categories)
        self.assertEqual('error', vector_indexer.handle_invalid)

        vector_indexer.set_input_col('test_input') \
            .set_output_col("test_output") \
            .set_max_categories(3) \
            .set_handle_invalid('skip')

        self.assertEqual('test_input', vector_indexer.input_col)
        self.assertEqual('test_output', vector_indexer.output_col)
        self.assertEqual(3, vector_indexer.max_categories)
        self.assertEqual('skip', vector_indexer.handle_invalid)

    def test_output_schema(self):
        vector_indexer = VectorIndexer()

        output = vector_indexer.fit(self.train_table).transform(self.predict_table)[0]

        self.assertEqual(
            ['input', 'output'],
            output.get_schema().get_field_names())

    def test_save_load_predict(self):
        vector_indexer = VectorIndexer().set_handle_invalid('keep')
        estimator_path = os.path.join(self.temp_dir, 'test_save_load_predict_vectorindexer')
        vector_indexer.save(estimator_path)
        vector_indexer = VectorIndexer.load(self.t_env, estimator_path)

        model = vector_indexer.fit(self.train_table)
        model_path = os.path.join(self.temp_dir, 'test_save_load_predict_vectorindexer_model')
        model.save(model_path)
        self.env.execute('save_model')
        model = VectorIndexerModel.load(self.t_env, model_path)

        output_table = model.transform(self.predict_table)[0]
        predicted_results = [result[1] for result in
                             self.t_env.to_data_stream(output_table).execute_and_collect()]

        predicted_results.sort(key=lambda x: x[1])
        self.expected_output.sort(key=lambda x: x[1])
        self.assertEqual(self.expected_output, predicted_results)
