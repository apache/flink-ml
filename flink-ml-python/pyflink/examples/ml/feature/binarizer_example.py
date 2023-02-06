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

# Simple program that creates a Binarizer instance and uses it for feature
# engineering.
#
# Before executing this program, please make sure you have followed Flink ML's
# quick start guideline to set up Flink ML and Flink environment. The guideline
# can be found at
#
# https://nightlies.apache.org/flink/flink-ml-docs-master/docs/try-flink-ml/quick-start/

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.binarizer import Binarizer
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (1,
         Vectors.dense(3, 4)),
        (2,
         Vectors.dense(6, 2))
    ],
        type_info=Types.ROW_NAMED(
            ['f0', 'f1'],
            [Types.INT(), DenseVectorTypeInfo()])))

# create an binarizer object and initialize its parameters
binarizer = Binarizer() \
    .set_input_cols('f0', 'f1') \
    .set_output_cols('of0', 'of1') \
    .set_thresholds(1.5, 3.5)

# use the binarizer for feature engineering
output = binarizer.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in binarizer.get_input_cols()]
output_values = [None for _ in binarizer.get_output_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(binarizer.get_input_cols())):
        input_values[i] = result[field_names.index(binarizer.get_input_cols()[i])]
        output_values[i] = result[field_names.index(binarizer.get_output_cols()[i])]
    print('Input Values: ' + str(input_values) + '\tOutput Values: ' + str(output_values))
