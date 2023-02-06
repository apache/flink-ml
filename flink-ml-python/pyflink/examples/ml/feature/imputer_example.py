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

# Simple program that creates an Imputer instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.imputer import Imputer
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input training and prediction data
train_data = t_env.from_data_stream(
    env.from_collection([
        (float('NaN'), 9.0,),
        (1.0, 9.0,),
        (1.5, 7.0,),
        (1.5, float('NaN'),),
        (4.0, 5.0,),
        (None, 4.0,),
    ],
        type_info=Types.ROW_NAMED(
            ['input1', 'input2'],
            [Types.DOUBLE(), Types.DOUBLE()])
    ))

# Creates an Imputer object and initializes its parameters.
imputer = Imputer()\
    .set_input_cols('input1', 'input2')\
    .set_output_cols('output1', 'output2')\
    .set_strategy('mean')\
    .set_missing_value(float('NaN'))

# Trains the Imputer Model.
model = imputer.fit(train_data)

# Uses the Imputer Model for predictions.
output = model.transform(train_data)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_values = []
    output_values = []
    for i in range(len(imputer.get_input_cols())):
        input_values.append(result[field_names.index(imputer.get_input_cols()[i])])
        output_values.append(result[field_names.index(imputer.get_output_cols()[i])])
    print('Input Values: ' + str(input_values) + '\tOutput Values: ' + str(output_values))
