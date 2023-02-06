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

# Simple program that creates a ChiSqTest instance and uses it for statistics.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.stats.chisqtest import ChiSqTest
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_table = t_env.from_data_stream(
    env.from_collection([
        (0., Vectors.dense(5, 1.)),
        (2., Vectors.dense(6, 2.)),
        (1., Vectors.dense(7, 2.)),
        (1., Vectors.dense(5, 4.)),
        (0., Vectors.dense(5, 1.)),
        (2., Vectors.dense(6, 2.)),
        (1., Vectors.dense(7, 2.)),
        (1., Vectors.dense(5, 4.)),
        (2., Vectors.dense(5, 1.)),
        (0., Vectors.dense(5, 2.)),
        (0., Vectors.dense(5, 2.)),
        (1., Vectors.dense(9, 4.)),
        (1., Vectors.dense(9, 3.))
    ],
        type_info=Types.ROW_NAMED(
            ['label', 'features'],
            [Types.DOUBLE(), DenseVectorTypeInfo()]))
)

# create a ChiSqTest object and initialize its parameters
chi_sq_test = ChiSqTest().set_features_col('features').set_label_col('label').set_flatten(True)

# use the ChiSqTest object for statistics
output = chi_sq_test.transform(input_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    print("Feature Index: %s\tP Value: %s\tDegree of Freedom: %s\tStatistics: %s" %
          (result[field_names.index('featureIndex')], result[field_names.index('pValue')],
           result[field_names.index('degreeOfFreedom')], result[field_names.index('statistic')]))
