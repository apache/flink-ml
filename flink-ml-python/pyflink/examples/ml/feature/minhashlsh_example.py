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

# Simple program that trains a MinHashLSH model and uses it for approximate nearest neighbors
# and similarity join.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.linalg import Vectors, SparseVectorTypeInfo
from pyflink.ml.feature.lsh import MinHashLSH

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates two datasets.
data_a = t_env.from_data_stream(
    env.from_collection([
        (0, Vectors.sparse(6, [0, 1, 2], [1., 1., 1.])),
        (1, Vectors.sparse(6, [2, 3, 4], [1., 1., 1.])),
        (2, Vectors.sparse(6, [0, 2, 4], [1., 1., 1.])),
    ], type_info=Types.ROW_NAMED(['id', 'vec'], [Types.INT(), SparseVectorTypeInfo()])))

data_b = t_env.from_data_stream(
    env.from_collection([
        (3, Vectors.sparse(6, [1, 3, 5], [1., 1., 1.])),
        (4, Vectors.sparse(6, [2, 3, 5], [1., 1., 1.])),
        (5, Vectors.sparse(6, [1, 2, 4], [1., 1., 1.])),
    ], type_info=Types.ROW_NAMED(['id', 'vec'], [Types.INT(), SparseVectorTypeInfo()])))

# Creates a MinHashLSH estimator object and initializes its parameters.
lsh = MinHashLSH() \
    .set_input_col('vec') \
    .set_output_col('hashes') \
    .set_seed(2022) \
    .set_num_hash_tables(5)

# Trains the MinHashLSH model.
model = lsh.fit(data_a)

# Uses the MinHashLSH model for transformation.
output = model.transform(data_a)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(lsh.get_input_col())]
    output_value = result[field_names.index(lsh.get_output_col())]
    print(f'Vector: {input_value} \tHash Values: {output_value}')

# Finds approximate nearest neighbors of the key.
key = Vectors.sparse(6, [1, 3], [1., 1.])
output = model.approx_nearest_neighbors(data_a, key, 2).select("id, distCol")
for result in t_env.to_data_stream(output).execute_and_collect():
    id_value = result[field_names.index("id")]
    dist_value = result[-1]
    print(f'ID: {id_value} \tDistance: {dist_value}')

# Approximately finds pairs from two datasets with distances smaller than the threshold.
output = model.approx_similarity_join(data_a, data_b, .6, "id")
for result in t_env.to_data_stream(output).execute_and_collect():
    id_a_value, id_b_value, dist_value = result
    print(f'ID from left: {id_a_value} \tID from right: {id_b_value} \t Distance: {dist_value}')
