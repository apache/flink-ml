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

# Simple program that creates a RobustScaler instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo

from pyflink.ml.feature.robustscaler import RobustScaler

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input training and prediction data.
train_data = t_env.from_data_stream(
    env.from_collection([
        (1, Vectors.dense(0.0, 0.0),),
        (2, Vectors.dense(1.0, -1.0),),
        (3, Vectors.dense(2.0, -2.0),),
        (4, Vectors.dense(3.0, -3.0),),
        (5, Vectors.dense(4.0, -4.0),),
        (6, Vectors.dense(5.0, -5.0),),
        (7, Vectors.dense(6.0, -6.0),),
        (8, Vectors.dense(7.0, -7.0),),
        (9, Vectors.dense(8.0, -8.0),),
    ],
        type_info=Types.ROW_NAMED(
            ['id', 'input'],
            [Types.INT(), DenseVectorTypeInfo()])
    ))

# Creates an RobustScaler object and initializes its parameters.
robust_scaler = RobustScaler()\
    .set_lower(0.25)\
    .set_upper(0.75)\
    .set_relative_error(0.001)\
    .set_with_scaling(True)\
    .set_with_centering(True)

# Trains the RobustScaler Model.
model = robust_scaler.fit(train_data)

# Uses the RobustScaler Model for predictions.
output = model.transform(train_data)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_index = field_names.index(robust_scaler.get_input_col())
    output_index = field_names.index(robust_scaler.get_output_col())
    print('Input Value: ' + str(result[input_index]) +
          '\tOutput Value: ' + str(result[output_index]))
