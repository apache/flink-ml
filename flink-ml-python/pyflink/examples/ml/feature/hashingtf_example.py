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

# Simple program that creates a HashingTF instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.lib.feature.hashingtf import HashingTF
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()

t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (['HashingTFTest', 'Hashing', 'Term', 'Frequency', 'Test'],),
        (['HashingTFTest', 'Hashing', 'Hashing', 'Test', 'Test'],),
    ],
        type_info=Types.ROW_NAMED(
            ["input", ],
            [Types.OBJECT_ARRAY(Types.STRING())])))

# Creates a HashingTF object and initializes its parameters.
hashing_tf = HashingTF() \
    .set_input_col('input') \
    .set_num_features(128) \
    .set_output_col('output')

# Uses the HashingTF object for feature transformations.
output = hashing_tf.transform(input_data_table)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(hashing_tf.get_input_col())]
    output_value = result[field_names.index(hashing_tf.get_output_col())]
    print('Input Value: ' + ' '.join(input_value) + '\tOutput Value: ' + str(output_value))
