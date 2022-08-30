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

# Simple program that trains a NaiveBayes model and uses it for classification.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.classification.naivebayes import NaiveBayes
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input training and prediction data
train_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([0, 0.]), 11.),
        (Vectors.dense([1, 0]), 10.),
        (Vectors.dense([1, 1.]), 10.),
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label'],
            [DenseVectorTypeInfo(), Types.DOUBLE()])))

predict_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([0, 1.]),),
        (Vectors.dense([0, 0.]),),
        (Vectors.dense([1, 0]),),
        (Vectors.dense([1, 1.]),),
    ],
        type_info=Types.ROW_NAMED(
            ['features'],
            [DenseVectorTypeInfo()])))

# create a naive bayes object and initialize its parameters
naive_bayes = NaiveBayes() \
    .set_smoothing(1.0) \
    .set_features_col('features') \
    .set_label_col('label') \
    .set_prediction_col('prediction') \
    .set_model_type('multinomial')

# train the naive bayes model
model = naive_bayes.fit(train_table)

# use the naive bayes model for predictions
output = model.transform(predict_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    features = result[field_names.index(naive_bayes.get_features_col())]
    prediction_result = result[field_names.index(naive_bayes.get_prediction_col())]
    print('Features: ' + str(features) + ' \tPrediction Result: ' + str(prediction_result))
