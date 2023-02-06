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

# Simple program that trains a KBinsDiscretizer model and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.kbinsdiscretizer import KBinsDiscretizer
from pyflink.table import StreamTableEnvironment

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input for training and prediction.
input_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(1, 10, 0),),
        (Vectors.dense(1, 10, 0),),
        (Vectors.dense(1, 10, 0),),
        (Vectors.dense(4, 10, 0),),
        (Vectors.dense(5, 10, 0),),
        (Vectors.dense(6, 10, 0),),
        (Vectors.dense(7, 10, 0),),
        (Vectors.dense(10, 10, 0),),
        (Vectors.dense(13, 10, 0),),
    ],
        type_info=Types.ROW_NAMED(
            ['input', ],
            [DenseVectorTypeInfo(), ])))

# Creates a KBinsDiscretizer object and initializes its parameters.
k_bins_discretizer = KBinsDiscretizer() \
    .set_input_col('input') \
    .set_output_col('output') \
    .set_num_bins(3) \
    .set_strategy('uniform')

# Trains the KBinsDiscretizer Model.
model = k_bins_discretizer.fit(input_table)

# Uses the KBinsDiscretizer Model for predictions.
output = model.transform(input_table)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    print('Input Value: ' + str(result[field_names.index(k_bins_discretizer.get_input_col())])
          + '\tOutput Value: ' +
          str(result[field_names.index(k_bins_discretizer.get_output_col())]))
