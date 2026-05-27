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

# Simple program that creates a Swing instance and gives recommendations for items.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.recommendation.als import Als

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_table = t_env.from_data_stream(
    env.from_collection([
        (1, 5, 0.1),
        (2, 8, 0.5),
        (3, 5, 0.8),
        (4, 7, 0.1),
        (1, 7, 0.7),
        (2, 5, 0.9),
        (3, 8, 0.1),
        (2, 6, 0.7),
        (2, 7, 0.4),
        (1, 8, 0.3),
        (4, 6, 0.4),
        (3, 7, 0.6),
        (1, 6, 0.5),
        (4, 8, 0.3)
    ],
        type_info=Types.ROW_NAMED(
            ['user', 'item', 'rating'],
            [Types.LONG(), Types.LONG(), Types.DOUBLE()])
    ))

test_table = t_env.from_data_stream(
    env.from_collection([
        (1, 6),
        (2, 7)
    ],
        type_info=Types.ROW_NAMED(
            ['user', 'item'],
            [Types.LONG(), Types.LONG()])
    ))

# Creates a als object and initialize its parameters.
als = Als()

# Transforms the data to Als algorithm result.
output_table = als.fit(input_table).transform(test_table)[0]

# Extracts and display the results.
field_names = output_table[0].get_schema().get_field_names()

results = t_env.to_data_stream(
    output_table[0]).execute_and_collect()

for result in results:
    user = result[field_names.index(als.get_user_col())]
    item = result[field_names.index(als.get_item_col())]
    score = result[field_names.index(als.get_prediction_col())]
    print(f'user: {user}, item : {item}, score: {score}')
