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

from pyflink.ml.recommendation.swing import Swing

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_table = t_env.from_data_stream(
    env.from_collection([
        (0, 10),
        (0, 11),
        (0, 12),
        (1, 13),
        (1, 12),
        (2, 10),
        (2, 11),
        (2, 12),
        (3, 13),
        (3, 12)
    ],
        type_info=Types.ROW_NAMED(
            ['user', 'item'],
            [Types.LONG(), Types.LONG()])
    ))

# Creates a swing object and initialize its parameters.
swing = Swing() \
    .set_item_col('item') \
    .set_user_col("user") \
    .set_min_user_behavior(1)

# Transforms the data to Swing algorithm result.
output_table = swing.transform(input_table)

# Extracts and display the results.
field_names = output_table[0].get_schema().get_field_names()

results = t_env.to_data_stream(
    output_table[0]).execute_and_collect()

for result in results:
    main_item = result[field_names.index(swing.get_item_col())]
    item_rank_score = result[1]
    print(f'item: {main_item}, top-k similar items: {item_rank_score}')
