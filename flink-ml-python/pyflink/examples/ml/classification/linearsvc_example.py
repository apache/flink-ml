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

# Simple program that trains a LinearSVC model and uses it for classification.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.classification.linearsvc import LinearSVC
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([1, 2, 3, 4]), 0., 1.),
        (Vectors.dense([2, 2, 3, 4]), 0., 2.),
        (Vectors.dense([3, 2, 3, 4]), 0., 3.),
        (Vectors.dense([4, 2, 3, 4]), 0., 4.),
        (Vectors.dense([5, 2, 3, 4]), 0., 5.),
        (Vectors.dense([11, 2, 3, 4]), 1., 1.),
        (Vectors.dense([12, 2, 3, 4]), 1., 2.),
        (Vectors.dense([13, 2, 3, 4]), 1., 3.),
        (Vectors.dense([14, 2, 3, 4]), 1., 4.),
        (Vectors.dense([15, 2, 3, 4]), 1., 5.),
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label', 'weight'],
            [DenseVectorTypeInfo(), Types.DOUBLE(), Types.DOUBLE()])
    ))

# create a linear svc object and initialize its parameters
linear_svc = LinearSVC().set_weight_col('weight')

# train the linear svc model
model = linear_svc.fit(input_table)

# use the linear svc model for predictions
output = model.transform(input_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    features = result[field_names.index(linear_svc.get_features_col())]
    expected_result = result[field_names.index(linear_svc.get_label_col())]
    prediction_result = result[field_names.index(linear_svc.get_prediction_col())]
    raw_prediction_result = result[field_names.index(linear_svc.get_raw_prediction_col())]
    print('Features: ' + str(features) + ' \tExpected Result: ' + str(expected_result)
          + ' \tPrediction Result: ' + str(prediction_result)
          + ' \tRaw Prediction Result: ' + str(raw_prediction_result))
