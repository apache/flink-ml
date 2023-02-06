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

# Simple program that converts a column of dense/sparse vectors into a column of double arrays.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.linalg import Vectors, VectorTypeInfo

from pyflink.ml.functions import vector_to_array
from pyflink.table.expressions import col

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input vector data
vectors = [
    (Vectors.dense(0.0, 0.0),),
    (Vectors.sparse(2, [1], [1.0]),),
]
input_table = t_env.from_data_stream(
    env.from_collection(
        vectors,
        type_info=Types.ROW_NAMED(
            ['vector'],
            [VectorTypeInfo()])
    ))

# convert each vector to a double array
output_table = input_table.select(vector_to_array(col('vector')).alias('array'))

# extract and display the results
output_values = [x for x in
                 t_env.to_data_stream(output_table).map(lambda r: r).execute_and_collect()]

output_values.sort(key=lambda x: x[0])

field_names = output_table.get_schema().get_field_names()
for i in range(len(output_values)):
    vector = vectors[i][0]
    double_array = output_values[i][field_names.index("array")]
    print("Input vector: %s \t output double array: %s" % (vector, double_array))
