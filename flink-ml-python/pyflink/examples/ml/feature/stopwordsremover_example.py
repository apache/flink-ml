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

# Simple program that creates a StopWordsRemover instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.stopwordsremover import StopWordsRemover
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_table = t_env.from_data_stream(
    env.from_collection([
        (["test", "test"],),
        (["a", "b", "c", "d"],),
        (["a", "the", "an"],),
        (["A", "The", "AN"],),
        ([None],),
        ([],),
    ],
        type_info=Types.ROW_NAMED(
            ['input'],
            [Types.OBJECT_ARRAY(Types.STRING())])))

# create a StopWordsRemover object and initialize its parameters
remover = StopWordsRemover().set_input_cols('input').set_output_cols('output')

# use the StopWordsRemover for feature engineering
output_table = remover.transform(input_table)[0]

# extract and display the results
field_names = output_table.get_schema().get_field_names()
for result in t_env.to_data_stream(output_table).execute_and_collect():
    input_value = result[field_names.index('input')]
    output_value = result[field_names.index('output')]
    print('Input Value: ' + str(input_value) + '\tOutput Value: ' + str(output_value))
