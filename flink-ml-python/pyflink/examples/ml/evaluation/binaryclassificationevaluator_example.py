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

# Simple program that creates a BinaryClassificationEvaluator instance and uses
# it for evaluation.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.evaluation.binaryclassification import BinaryClassificationEvaluator
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_table = t_env.from_data_stream(
    env.from_collection([
        (1.0, Vectors.dense(0.1, 0.9)),
        (1.0, Vectors.dense(0.2, 0.8)),
        (1.0, Vectors.dense(0.3, 0.7)),
        (0.0, Vectors.dense(0.25, 0.75)),
        (0.0, Vectors.dense(0.4, 0.6)),
        (1.0, Vectors.dense(0.35, 0.65)),
        (1.0, Vectors.dense(0.45, 0.55)),
        (0.0, Vectors.dense(0.6, 0.4)),
        (0.0, Vectors.dense(0.7, 0.3)),
        (1.0, Vectors.dense(0.65, 0.35)),
        (0.0, Vectors.dense(0.8, 0.2)),
        (1.0, Vectors.dense(0.9, 0.1))
    ],
        type_info=Types.ROW_NAMED(
            ['label', 'rawPrediction'],
            [Types.DOUBLE(), DenseVectorTypeInfo()]))
)

# create a binary classification evaluator object and initialize its parameters
evaluator = BinaryClassificationEvaluator() \
    .set_metrics_names('areaUnderPR', 'ks', 'areaUnderROC')

# use the binary classification evaluator model for evaluations
output = evaluator.transform(input_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
result = t_env.to_data_stream(output).execute_and_collect().next()
print('Area under the precision-recall curve: '
      + str(result[field_names.index('areaUnderPR')]))
print('Area under the receiver operating characteristic curve: '
      + str(result[field_names.index('areaUnderROC')]))
print('Kolmogorov-Smirnov value: '
      + str(result[field_names.index('ks')]))
