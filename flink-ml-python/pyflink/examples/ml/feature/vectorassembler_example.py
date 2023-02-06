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

# Simple program that creates a VectorAssembler instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo, SparseVectorTypeInfo
from pyflink.ml.feature.vectorassembler import VectorAssembler
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(2.1, 3.1),
         1.0,
         Vectors.sparse(5, [3], [1.0])),
        (Vectors.dense(2.1, 3.1),
         1.0,
         Vectors.sparse(5, [1, 2, 3, 4],
                        [1.0, 2.0, 3.0, 4.0])),
    ],
        type_info=Types.ROW_NAMED(
            ['vec', 'num', 'sparse_vec'],
            [DenseVectorTypeInfo(), Types.DOUBLE(), SparseVectorTypeInfo()])))

# create a vector assembler object and initialize its parameters
vector_assembler = VectorAssembler() \
    .set_input_cols('vec', 'num', 'sparse_vec') \
    .set_output_col('assembled_vec') \
    .set_input_sizes(2, 1, 5) \
    .set_handle_invalid('keep')

# use the vector assembler for feature engineering
output = vector_assembler.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in vector_assembler.get_input_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(vector_assembler.get_input_cols())):
        input_values[i] = result[field_names.index(vector_assembler.get_input_cols()[i])]
    output_value = result[field_names.index(vector_assembler.get_output_col())]
    print('Input Values: ' + str(input_values) + '\tOutput Value: ' + str(output_value))
