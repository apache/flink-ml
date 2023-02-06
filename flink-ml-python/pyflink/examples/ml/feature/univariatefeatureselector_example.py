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

# Simple program that creates a UnivariateFeatureSelector instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.univariatefeatureselector import UnivariateFeatureSelector
from pyflink.table import StreamTableEnvironment

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo

env = StreamExecutionEnvironment.get_execution_environment()

t_env = StreamTableEnvironment.create(env)

# Generates input training and prediction data.
input_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(1.7, 4.4, 7.6, 5.8, 9.6, 2.3), 3.0,),
        (Vectors.dense(8.8, 7.3, 5.7, 7.3, 2.2, 4.1), 2.0,),
        (Vectors.dense(1.2, 9.5, 2.5, 3.1, 8.7, 2.5), 1.0,),
        (Vectors.dense(3.7, 9.2, 6.1, 4.1, 7.5, 3.8), 2.0,),
        (Vectors.dense(8.9, 5.2, 7.8, 8.3, 5.2, 3.0), 4.0,),
        (Vectors.dense(7.9, 8.5, 9.2, 4.0, 9.4, 2.1), 4.0,),
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label'],
            [DenseVectorTypeInfo(), Types.FLOAT()])
    ))

# Creates an UnivariateFeatureSelector object and initializes its parameters.
univariate_feature_selector = UnivariateFeatureSelector() \
    .set_features_col('features') \
    .set_label_col('label') \
    .set_feature_type('continuous') \
    .set_label_type('categorical') \
    .set_selection_threshold(1)

# Trains the UnivariateFeatureSelector Model.
model = univariate_feature_selector.fit(input_table)

# Uses the UnivariateFeatureSelector Model for predictions.
output = model.transform(input_table)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_index = field_names.index(univariate_feature_selector.get_features_col())
    output_index = field_names.index(univariate_feature_selector.get_output_col())
    print('Input Value: ' + str(result[input_index]) +
          '\tOutput Value: ' + str(result[output_index]))
