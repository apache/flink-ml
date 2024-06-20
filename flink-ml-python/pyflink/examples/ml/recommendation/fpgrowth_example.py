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

# Simple program that creates a fpgrowth instance and gives recommendations for items.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.recommendation.fpgrowth import FPGrowth

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_table = t_env.from_data_stream(
    env.from_collection([
        ("A,B,C,D",),
        ("B,C,E",),
        ("A,B,C,E",),
        ("B,D,E",),
        ("A,B,C,D",)
    ],
        type_info=Types.ROW_NAMED(
            ['items'],
            [Types.STRING()])
    ))

# Creates a fpgrowth object and initialize its parameters.
fpg = FPGrowth().set_min_support(0.6)

# Transforms the data to fpgrowth algorithm result.
output_table = fpg.transform(input_table)

# Extracts and display the results.
pattern_result_names = output_table[0].get_schema().get_field_names()
rule_result_names = output_table[1].get_schema().get_field_names()

patterns = t_env.to_data_stream(output_table[0]).execute_and_collect()
rules = t_env.to_data_stream(output_table[1]).execute_and_collect()

print("|\t"+"\t|\t".join(pattern_result_names)+"\t|")
for result in patterns:
    print(f'|\t{result[0]}\t|\t{result[1]}\t|\t{result[2]}\t|')
print("|\t"+" | ".join(rule_result_names)+"\t|")
for result in rules:
    print(f'|\t{result[0]}\t|\t{result[1]}\t|\t{result[2]}\t|\t{result[3]}'
          + f'\t|\t{result[4]}\t|\t{result[5]}\t|')
