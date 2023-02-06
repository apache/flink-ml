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

# Simple program that trains a VarianceThresholdSelector model and uses it for feature
# selection.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.variancethresholdselector import VarianceThresholdSelector
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input training and prediction data
train_data = t_env.from_data_stream(
    env.from_collection([
        (1, Vectors.dense(5.0, 7.0, 0.0, 7.0, 6.0, 0.0),),
        (2, Vectors.dense(0.0, 9.0, 6.0, 0.0, 5.0, 9.0),),
        (3, Vectors.dense(0.0, 9.0, 3.0, 0.0, 5.0, 5.0),),
        (4, Vectors.dense(1.0, 9.0, 8.0, 5.0, 7.0, 4.0),),
        (5, Vectors.dense(9.0, 8.0, 6.0, 5.0, 4.0, 4.0),),
        (6, Vectors.dense(6.0, 9.0, 7.0, 0.0, 2.0, 0.0),),
    ],
        type_info=Types.ROW_NAMED(
            ['id', 'input'],
            [Types.INT(), DenseVectorTypeInfo()])
    ))

# create a VarianceThresholdSelector object and initialize its parameters
threshold = 8.0
variance_thread_selector = VarianceThresholdSelector()\
    .set_input_col("input")\
    .set_variance_threshold(threshold)

# train the VarianceThresholdSelector model
model = variance_thread_selector.fit(train_data)

# use the VarianceThresholdSelector model for predictions
output = model.transform(train_data)[0]

# extract and display the results
print("Variance Threshold: " + str(threshold))
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(variance_thread_selector.get_input_col())]
    output_value = result[field_names.index(variance_thread_selector.get_output_col())]
    print('Input Values: ' + str(input_value) + ' \tOutput Values: ' + str(output_value))
