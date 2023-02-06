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

# Simple program that trains a OneHotEncoder model and uses it for feature
# engineering.

from pyflink.common import Row
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.onehotencoder import OneHotEncoder
from pyflink.table import StreamTableEnvironment, DataTypes

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input training and prediction data
train_table = t_env.from_elements(
    [Row(0.0), Row(1.0), Row(2.0), Row(0.0)],
    DataTypes.ROW([
        DataTypes.FIELD('input', DataTypes.DOUBLE())
    ]))

predict_table = t_env.from_elements(
    [Row(0.0), Row(1.0), Row(2.0)],
    DataTypes.ROW([
        DataTypes.FIELD('input', DataTypes.DOUBLE())
    ]))

# create a one-hot-encoder object and initialize its parameters
one_hot_encoder = OneHotEncoder().set_input_cols('input').set_output_cols('output')

# train the one-hot-encoder model
model = one_hot_encoder.fit(train_table)

# use the one-hot-encoder model for predictions
output = model.transform(predict_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(one_hot_encoder.get_input_cols()[0])]
    output_value = result[field_names.index(one_hot_encoder.get_output_cols()[0])]
    print('Input Value: ' + str(input_value) + ' \tOutput Value: ' + str(output_value))
