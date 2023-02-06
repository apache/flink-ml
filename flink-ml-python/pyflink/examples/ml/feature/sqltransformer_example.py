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

# Simple program that creates a SQLTransformer instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.sqltransformer import SQLTransformer
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()

t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (0, 1.0, 3.0),
        (2, 2.0, 5.0),
    ],
        type_info=Types.ROW_NAMED(
            ['id', 'v1', 'v2'],
            [Types.INT(), Types.DOUBLE(), Types.DOUBLE()])))

# Creates a SQLTransformer object and initializes its parameters.
sql_transformer = SQLTransformer() \
    .set_statement('SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__')

# Uses the SQLTransformer object for feature transformations.
output_table = sql_transformer.transform(input_data_table)[0]

# Extracts and displays the results.
output_table.execute().print()
