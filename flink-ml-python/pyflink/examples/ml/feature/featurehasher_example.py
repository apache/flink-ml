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

# Simple program that creates a FeatureHasher instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.featurehasher import FeatureHasher
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (0, 'a', 1.0, True),
        (1, 'c', 1.0, False),
    ],
        type_info=Types.ROW_NAMED(
            ['id', 'f0', 'f1', 'f2'],
            [Types.INT(), Types.STRING(), Types.DOUBLE(), Types.BOOLEAN()])))

# create a feature hasher object and initialize its parameters
feature_hasher = FeatureHasher() \
    .set_input_cols('f0', 'f1', 'f2') \
    .set_categorical_cols('f0', 'f2') \
    .set_output_col('vec') \
    .set_num_features(1000)

# use the feature hasher for feature engineering
output = feature_hasher.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in feature_hasher.get_input_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(feature_hasher.get_input_cols())):
        input_values[i] = result[field_names.index(feature_hasher.get_input_cols()[i])]
    output_value = result[field_names.index(feature_hasher.get_output_col())]
    print('Input Values: ' + str(input_values) + '\tOutput Value: ' + str(output_value))
