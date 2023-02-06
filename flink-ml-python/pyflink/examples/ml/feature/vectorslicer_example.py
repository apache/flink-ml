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

# Simple program that creates a VectorSlicer instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.vectorslicer import VectorSlicer
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (1, Vectors.dense(2.1, 3.1, 1.2, 2.1)),
        (2, Vectors.dense(2.3, 2.1, 1.3, 1.2)),
    ],
        type_info=Types.ROW_NAMED(
            ['id', 'vec'],
            [Types.INT(), DenseVectorTypeInfo()])))

# create a vector slicer object and initialize its parameters
vector_slicer = VectorSlicer() \
    .set_input_col('vec') \
    .set_indices(1, 2, 3) \
    .set_output_col('sub_vec')

# use the vector slicer model for feature engineering
output = vector_slicer.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(vector_slicer.get_input_col())]
    output_value = result[field_names.index(vector_slicer.get_output_col())]
    print('Input Value: ' + str(input_value) + '\tOutput Value: ' + str(output_value))
