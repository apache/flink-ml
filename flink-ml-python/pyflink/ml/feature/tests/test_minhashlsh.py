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
import functools
import os
from typing import List

from pyflink.common import Row, Types
from pyflink.java_gateway import get_gateway
from pyflink.ml.linalg import Vectors, SparseVectorTypeInfo, DenseVector
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.lsh import MinHashLSH, MinHashLSHModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase
from pyflink.table import Table


class MinHashLSHTest(PyFlinkMLTestCase):
    def setUp(self):
        super(MinHashLSHTest, self).setUp()
        self.data = self.t_env.from_data_stream(
            self.env.from_collection([
                (0, Vectors.sparse(6, [0, 1, 2], [1., 1., 1.])),
                (1, Vectors.sparse(6, [2, 3, 4], [1., 1., 1.])),
                (2, Vectors.sparse(6, [0, 2, 4], [1., 1., 1.])),
            ],
                type_info=Types.ROW_NAMED(
                    ['id', 'vec'],
                    [Types.INT(), SparseVectorTypeInfo()])))

        self.expected = [
            Row([
                Vectors.dense(1.73046954E8, 1.57275425E8, 6.90717571E8),
                Vectors.dense(5.02301169E8, 7.967141E8, 4.06089319E8),
                Vectors.dense(2.83652171E8, 1.97714719E8, 6.04731316E8),
                Vectors.dense(5.2181506E8, 6.36933726E8, 6.13894128E8),
                Vectors.dense(3.04301769E8, 1.113672955E9, 6.1388711E8),
            ]),
            Row([
                Vectors.dense(1.73046954E8, 1.57275425E8, 6.7798584E7),
                Vectors.dense(6.38582806E8, 1.78703694E8, 4.06089319E8),
                Vectors.dense(6.232638E8, 9.28867E7, 9.92010642E8),
                Vectors.dense(2.461064E8, 1.12787481E8, 1.92180297E8),
                Vectors.dense(2.38162496E8, 1.552933319E9, 2.77995137E8),
            ]),
            Row([
                Vectors.dense(1.73046954E8, 1.57275425E8, 6.90717571E8),
                Vectors.dense(1.453197722E9, 7.967141E8, 4.06089319E8),
                Vectors.dense(6.232638E8, 1.97714719E8, 6.04731316E8),
                Vectors.dense(2.461064E8, 1.12787481E8, 1.92180297E8),
                Vectors.dense(1.224130231E9, 1.113672955E9, 2.77995137E8),
            ])]

    def test_param(self):
        lsh = MinHashLSH()
        self.assertEqual('input', lsh.input_col)
        self.assertEqual('output', lsh.output_col)
        self.assertEqual(-1229568175, lsh.seed)
        self.assertEqual(1, lsh.num_hash_tables)
        self.assertEqual(1, lsh.num_hash_functions_per_table)

        lsh.set_input_col('test_input') \
            .set_output_col('test_output') \
            .set_seed(2022) \
            .set_num_hash_tables(3) \
            .set_num_hash_functions_per_table(4)

        self.assertEqual('test_input', lsh.input_col)
        self.assertEqual('test_output', lsh.output_col)
        self.assertEqual(2022, lsh.seed)
        self.assertEqual(3, lsh.num_hash_tables)
        self.assertEqual(4, lsh.num_hash_functions_per_table)

    def test_output_schema(self):
        lsh = MinHashLSH() \
            .set_input_col('vec') \
            .set_output_col('hashes') \
            .set_seed(2022) \
            .set_num_hash_tables(5) \
            .set_num_hash_functions_per_table(3)
        model = lsh.fit(self.data)
        output = model.transform(self.data)[0]
        self.assertEqual(['id', 'vec', 'hashes'], output.get_schema().get_field_names())

    def test_fit_and_transform(self):
        lsh = MinHashLSH() \
            .set_input_col('vec') \
            .set_output_col('hashes') \
            .set_seed(2022) \
            .set_num_hash_tables(5) \
            .set_num_hash_functions_per_table(3)
        model = lsh.fit(self.data)
        output = model.transform(self.data)[0].select("hashes")
        self.verify_output_hashes(output, self.expected)

    def test_estimator_save_load_transform(self):
        path = os.path.join(self.temp_dir, 'test_estimator_save_load_transform_minhashlsh')
        lsh = MinHashLSH() \
            .set_input_col('vec') \
            .set_output_col('hashes') \
            .set_seed(2022) \
            .set_num_hash_tables(5) \
            .set_num_hash_functions_per_table(3)
        lsh.save(path)
        lsh = MinHashLSH.load(self.t_env, path)
        model = lsh.fit(self.data)
        output = model.transform(self.data)[0].select(lsh.output_col)
        self.verify_output_hashes(output, self.expected)

    def test_model_save_load_transform(self):
        path = os.path.join(self.temp_dir, 'test_model_save_load_transform_minhashlsh')
        lsh = MinHashLSH() \
            .set_input_col('vec') \
            .set_output_col('hashes') \
            .set_seed(2022) \
            .set_num_hash_tables(5) \
            .set_num_hash_functions_per_table(3)
        model = lsh.fit(self.data)
        model.save(path)
        self.env.execute('save_model')
        model = MinHashLSHModel.load(self.t_env, path)
        output = model.transform(self.data)[0].select(lsh.output_col)
        self.verify_output_hashes(output, self.expected)

    def test_get_model_data(self):
        lsh = MinHashLSH() \
            .set_input_col('vec') \
            .set_output_col('hashes') \
            .set_seed(2022) \
            .set_num_hash_tables(5) \
            .set_num_hash_functions_per_table(3)
        model = lsh.fit(self.data)
        model_data_table: Table = model.get_model_data()[0]
        self.assertEqual(
            ['numHashTables', 'numHashFunctionsPerTable', 'randCoefficientA', 'randCoefficientB'],
            model_data_table.get_schema().get_field_names())

        model_data_row = list(self.t_env.to_data_stream(model_data_table).execute_and_collect())[0]
        self.assertEqual(lsh.num_hash_tables, model_data_row[0])
        self.assertEqual(lsh.num_hash_functions_per_table, model_data_row[1])
        self.assertEqual(lsh.num_hash_tables * lsh.num_hash_functions_per_table,
                         len(model_data_row[2]))
        self.assertEqual(lsh.num_hash_tables * lsh.num_hash_functions_per_table,
                         len(model_data_row[3]))

    def test_set_model_data(self):
        lsh = MinHashLSH() \
            .set_input_col('vec') \
            .set_output_col('hashes') \
            .set_seed(2022) \
            .set_num_hash_tables(5) \
            .set_num_hash_functions_per_table(3)
        model_a = lsh.fit(self.data)
        model_data_table = model_a.get_model_data()[0]
        model_b: MinHashLSHModel = MinHashLSHModel().set_model_data(model_data_table)
        self.update_existing_params(model_b, model_a)
        output = model_b.transform(self.data)[0].select(lsh.output_col)
        self.verify_output_hashes(output, self.expected)

    def test_approx_nearest_neighbors(self):
        lsh = MinHashLSH() \
            .set_input_col('vec') \
            .set_output_col('hashes') \
            .set_seed(2022) \
            .set_num_hash_tables(5) \
            .set_num_hash_functions_per_table(1)
        expected = [
            Row(0, 0.75),
            Row(1, 0.75),
        ]

        model: MinHashLSHModel = lsh.fit(self.data)
        key = Vectors.sparse(6, [1, 3], [1., 1.])
        output = model.approx_nearest_neighbors(self.data, key, 2).select("id, distCol")
        actual_result = [r for r in self.t_env.to_data_stream(output).execute_and_collect()]
        actual_result.sort(key=lambda r: r[0])
        self.assertEqual(expected, actual_result)

    def test_approx_nearest_neighbors_dense_vector(self):
        lsh = MinHashLSH() \
            .set_input_col('vec') \
            .set_output_col('hashes') \
            .set_seed(2022) \
            .set_num_hash_tables(5) \
            .set_num_hash_functions_per_table(1)
        expected = [
            Row(0, 0.75),
            Row(1, 0.75),
        ]

        model: MinHashLSHModel = lsh.fit(self.data)
        key = Vectors.dense([0., 1., 0., 1., 0., 0.])
        output = model.approx_nearest_neighbors(self.data, key, 2).select("id, distCol")
        actual_result = [r for r in self.t_env.to_data_stream(output).execute_and_collect()]
        actual_result.sort(key=lambda r: r[0])
        self.assertEqual(expected, actual_result)

    def test_approx_similarity_join(self):
        lsh = MinHashLSH() \
            .set_input_col('vec') \
            .set_output_col('hashes') \
            .set_seed(2022) \
            .set_num_hash_tables(5) \
            .set_num_hash_functions_per_table(1)
        data_a = self.data
        model: MinHashLSHModel = lsh.fit(data_a)
        data_b = self.t_env.from_data_stream(
            self.env.from_collection([
                (3, Vectors.sparse(6, [1, 3, 5], [1., 1., 1.])),
                (4, Vectors.sparse(6, [2, 3, 5], [1., 1., 1.])),
                (5, Vectors.sparse(6, [1, 2, 4], [1., 1., 1.])),
            ], type_info=Types.ROW_NAMED(['id', 'vec'], [Types.INT(), SparseVectorTypeInfo()])))
        expected = [
            Row(1, 4, .5),
            Row(0, 5, .5),
            Row(1, 5, .5),
            Row(2, 5, .5)
        ]
        output = model.approx_similarity_join(data_a, data_b, .6, "id")
        actual_result = [r for r in self.t_env.to_data_stream(output).execute_and_collect()]

        expected.sort(key=lambda r: (r[0], r[1]))
        actual_result.sort(key=lambda r: (r[0], r[1]))
        self.assertEqual(expected, actual_result)

    @classmethod
    def update_existing_params(cls, target: JavaWithParams, source: JavaWithParams):
        get_gateway().jvm.org.apache.flink.ml.util.ParamUtils \
            .updateExistingParams(target._java_obj, source._java_obj.getParamMap())

    @classmethod
    def dense_vector_comparator(cls, dv0: DenseVector, dv1: DenseVector):
        if dv0.size() != dv1.size():
            return dv0.size() - dv1.size()
        for e0, e1 in zip(dv0.values, dv1.values):
            if e0 != e1:
                return 1 if e0 > e1 else -1
        return 0

    @classmethod
    def dense_vector_array_comparator(cls, dvs0: List[DenseVector], dvs1: List[DenseVector]):
        if len(dvs0) != len(dvs1):
            return len(dvs0) - len(dvs1)
        for dv0, dv1 in zip(dvs0, dvs1):
            cmp = cls.dense_vector_comparator(dv0, dv1)
            if cmp != 0:
                return cmp
        return 0

    def verify_output_hashes(self, output: Table, expected_result: List[Row]):
        actual_result = [r for r in
                         self.t_env.to_data_stream(output).execute_and_collect()]
        actual_result.sort(
            key=lambda x: functools.cmp_to_key(self.dense_vector_array_comparator)(x[0]))
        expected_result.sort(
            key=lambda x: functools.cmp_to_key(self.dense_vector_array_comparator)(x[0]))
        for r0, r1 in zip(actual_result, expected_result):
            self.assertEqual(0, self.dense_vector_array_comparator(r0[0], r1[0]))
