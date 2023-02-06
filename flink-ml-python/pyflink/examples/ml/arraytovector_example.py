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

# Simple program that converts a column of double arrays into a column of dense vectors.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.functions import array_to_vector
from pyflink.table import StreamTableEnvironment
from pyflink.table.expressions import col

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input double array data
double_arrays = [
    ([0.0, 0.0],),
    ([0.0, 1.0],),
]
input_table = t_env.from_data_stream(
    env.from_collection(
        double_arrays,
        type_info=Types.ROW_NAMED(
            ['array'],
            [Types.PRIMITIVE_ARRAY(Types.DOUBLE())])
    ))

# convert each double array to a dense vector
output_table = input_table.select(array_to_vector(col('array')).alias('vector'))

# extract and display the results
field_names = output_table.get_schema().get_field_names()

output_values = [x[field_names.index('vector')] for x in
                 t_env.to_data_stream(output_table).execute_and_collect()]

output_values.sort(key=lambda x: x.get(1))

for i in range(len(output_values)):
    double_array = double_arrays[i][0]
    vector = output_values[i]
    print("Input double array: %s \t output vector: %s" % (double_array, vector))
