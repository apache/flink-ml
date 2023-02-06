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

# Simple program that creates a Interaction instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.interaction import Interaction
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (1,
         Vectors.dense(1, 2),
         Vectors.dense(3, 4)),
        (2,
         Vectors.dense(2, 8),
         Vectors.dense(3, 4))
    ],
        type_info=Types.ROW_NAMED(
            ['f0', 'f1', 'f2'],
            [Types.INT(), DenseVectorTypeInfo(), DenseVectorTypeInfo()])))

# create an interaction object and initialize its parameters
interaction = Interaction() \
    .set_input_cols('f0', 'f1', 'f2') \
    .set_output_col('interaction_vec')

# use the interaction for feature engineering
output = interaction.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in interaction.get_input_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(interaction.get_input_cols())):
        input_values[i] = result[field_names.index(interaction.get_input_cols()[i])]
    output_value = result[field_names.index(interaction.get_output_col())]
    print('Input Values: ' + str(input_values) + '\tOutput Value: ' + str(output_value))
