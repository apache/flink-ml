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

# Simple program that creates an IndexToStringModel instance and uses it
# for feature engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.stringindexer import IndexToStringModel
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
predict_table = t_env.from_data_stream(
    env.from_collection([
        (0, 3),
        (1, 2),
    ],
        type_info=Types.ROW_NAMED(
            ['input_col1', 'input_col2'],
            [Types.INT(), Types.INT()])
    ))

# create an index-to-string model and initialize its parameters and model data
model_data_table = t_env.from_data_stream(
    env.from_collection([
        ([['a', 'b', 'c', 'd'], [-1., 0., 1., 2.]],),
    ],
        type_info=Types.ROW_NAMED(
            ['stringArrays'],
            [Types.OBJECT_ARRAY(Types.OBJECT_ARRAY(Types.STRING()))])
    ))

model = IndexToStringModel() \
    .set_input_cols('input_col1', 'input_col2') \
    .set_output_cols('output_col1', 'output_col2') \
    .set_model_data(model_data_table)

# use the index-to-string model for feature engineering
output = model.transform(predict_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in model.get_input_cols()]
output_values = [None for _ in model.get_input_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(model.get_input_cols())):
        input_values[i] = result[field_names.index(model.get_input_cols()[i])]
        output_values[i] = result[field_names.index(model.get_output_cols()[i])]
    print('Input Values: ' + str(input_values) + '\tOutput Values: ' + str(output_values))
