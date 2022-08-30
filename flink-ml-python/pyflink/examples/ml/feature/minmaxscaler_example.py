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

# Simple program that trains a MinMaxScaler model and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.feature.minmaxscaler import MinMaxScaler
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input training and prediction data
train_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(0.0, 3.0),),
        (Vectors.dense(2.1, 0.0),),
        (Vectors.dense(4.1, 5.1),),
        (Vectors.dense(6.1, 8.1),),
        (Vectors.dense(200, 400),),
    ],
        type_info=Types.ROW_NAMED(
            ['input'],
            [DenseVectorTypeInfo()])
    ))

predict_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(150.0, 90.0),),
        (Vectors.dense(50.0, 40.0),),
        (Vectors.dense(100.0, 50.0),),
    ],
        type_info=Types.ROW_NAMED(
            ['input'],
            [DenseVectorTypeInfo()])
    ))

# create a min-max-scaler object and initialize its parameters
min_max_scaler = MinMaxScaler()

# train the min-max-scaler model
model = min_max_scaler.fit(train_data)

# use the min-max-scaler model for predictions
output = model.transform(predict_data)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(min_max_scaler.get_input_col())]
    output_value = result[field_names.index(min_max_scaler.get_output_col())]
    print('Input Value: ' + str(input_value) + ' \tOutput Value: ' + str(output_value))
