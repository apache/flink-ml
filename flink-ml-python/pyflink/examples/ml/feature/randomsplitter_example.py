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

# Simple program that creates a RandomSplitter instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.randomsplitter import RandomSplitter
from pyflink.table import StreamTableEnvironment

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input table.
input_table = t_env.from_data_stream(
    env.from_collection([
        (1, 10, 0),
        (1, 10, 0),
        (1, 10, 0),
        (4, 10, 0),
        (5, 10, 0),
        (6, 10, 0),
        (7, 10, 0),
        (10, 10, 0),
        (13, 10, 0)
    ],
        type_info=Types.ROW_NAMED(
            ['f0', 'f1', "f2"],
            [Types.INT(), Types.INT(), Types.INT()])))

# Creates a RandomSplitter object and initializes its parameters.
splitter = RandomSplitter().set_weights(4.0, 6.0).set_seed(0)

# Uses the RandomSplitter to split the dataset.
output = splitter.transform(input_table)

# Extracts and displays the results.
print("Split Result 1 (40%)")
for result in t_env.to_data_stream(output[0]).execute_and_collect():
    print(str(result))

print("Split Result 2 (60%)")
for result in t_env.to_data_stream(output[1]).execute_and_collect():
    print(str(result))
