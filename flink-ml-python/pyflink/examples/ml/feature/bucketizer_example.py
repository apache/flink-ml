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

# Simple program that creates a Bucketizer instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.bucketizer import Bucketizer
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data = t_env.from_data_stream(
    env.from_collection([
        (-0.5, 0.0, 1.0, 0.0),
    ],
        type_info=Types.ROW_NAMED(
            ['f1', 'f2', 'f3', 'f4'],
            [Types.DOUBLE(), Types.DOUBLE(), Types.DOUBLE(), Types.DOUBLE()])
    ))

# create a bucketizer object and initialize its parameters
splits_array = [
    [-0.5, 0.0, 0.5],
    [-1.0, 0.0, 2.0],
    [float('-inf'), 10.0, float('inf')],
    [float('-inf'), 1.5, float('inf')],
]

bucketizer = Bucketizer() \
    .set_input_cols('f1', 'f2', 'f3', 'f4') \
    .set_output_cols('o1', 'o2', 'o3', 'o4') \
    .set_splits_array(splits_array)

# use the bucketizer model for feature engineering
output = bucketizer.transform(input_data)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in bucketizer.get_input_cols()]
output_values = [None for _ in bucketizer.get_input_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(bucketizer.get_input_cols())):
        input_values[i] = result[field_names.index(bucketizer.get_input_cols()[i])]
        output_values[i] = result[field_names.index(bucketizer.get_output_cols()[i])]
    print('Input Values: ' + str(input_values) + '\tOutput Values: ' + str(output_values))
