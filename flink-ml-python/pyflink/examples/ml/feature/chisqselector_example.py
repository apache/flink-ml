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

# Simple program that trains a ChiSqSelector model and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, VectorTypeInfo
from pyflink.ml.lib.feature.chisqselector import ChiSqSelector
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_table = t_env.from_data_stream(
    env.from_collection([
        (0.0, Vectors.sparse(6, [0, 1, 3, 4], [6.0, 7.0, 7.0, 6.0]),),
        (0.0, Vectors.sparse(6, [1, 2, 4, 5], [9.0, 6.0, 5.0, 9.0]),),
        (0.0, Vectors.sparse(6, [1, 2, 4, 5], [9.0, 3.0, 5.0, 5.0]),),
        (1.0, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0),),
        (2.0, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0),),
        (2.0, Vectors.dense(8.0, 9.0, 6.0, 4.0, 0.0, 0.0),),
    ],
        type_info=Types.ROW_NAMED(
            ['label', 'features'],
            [Types.DOUBLE(), VectorTypeInfo()])))

# create a ChiSqSelector object and initialize its parameters
selector = ChiSqSelector()

# train the ChiSqSelector model
model = selector.fit(input_table)

# use the ChiSqSelector model for predictions
output_table = model.transform(input_table)[0]

# extract and display the results
field_names = output_table.get_schema().get_field_names()
for result in t_env.to_data_stream(output_table).execute_and_collect():
    input_value = result[field_names.index(selector.get_features_col())]
    output_value = result[field_names.index(selector.get_output_col())]
    print('Input Value: ' + str(input_value) + ' \tOutput Value: ' + str(output_value))
