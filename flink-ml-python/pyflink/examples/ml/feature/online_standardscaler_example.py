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

# Simple program that trains an OnlineStandardScaler model and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.common.time import Time, Instant
from pyflink.java_gateway import get_gateway
from pyflink.table import Schema
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
from pyflink.table.expressions import col

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.onlinestandardscaler import OnlineStandardScaler
from pyflink.ml.common.window import EventTimeTumblingWindows

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input data.
dense_vector_serializer = get_gateway().jvm.org.apache.flink.table.types.logical.RawType(
    get_gateway().jvm.org.apache.flink.ml.linalg.DenseVector(0).getClass(),
    get_gateway().jvm.org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer()
).getSerializerString()

schema = Schema.new_builder() \
    .column("ts", "TIMESTAMP_LTZ(3)") \
    .column("input", "RAW('org.apache.flink.ml.linalg.DenseVector', '{serializer}')"
            .format(serializer=dense_vector_serializer)) \
    .watermark("ts", "ts - INTERVAL '1' SECOND") \
    .build()

input_data = t_env.from_data_stream(
    env.from_collection([
        (Instant.of_epoch_milli(0), Vectors.dense(-2.5, 9, 1),),
        (Instant.of_epoch_milli(1000), Vectors.dense(1.4, -5, 1),),
        (Instant.of_epoch_milli(2000), Vectors.dense(2, -1, -2),),
        (Instant.of_epoch_milli(6000), Vectors.dense(0.7, 3, 1),),
        (Instant.of_epoch_milli(7000), Vectors.dense(0, 1, 1),),
        (Instant.of_epoch_milli(8000), Vectors.dense(0.5, 0, -2),),
        (Instant.of_epoch_milli(9000), Vectors.dense(0.4, 1, 1),),
        (Instant.of_epoch_milli(10000), Vectors.dense(0.3, 2, 1),),
        (Instant.of_epoch_milli(11000), Vectors.dense(0.5, 1, -2),)
    ],
        type_info=Types.ROW_NAMED(
            ['ts', 'input'],
            [Types.INSTANT(), DenseVectorTypeInfo()])),
    schema)

# Creates an online standard-scaler object and initialize its parameters.
standard_scaler = OnlineStandardScaler() \
    .set_windows(EventTimeTumblingWindows.of(Time.milliseconds(3000))) \
    .set_max_allowed_model_delay_ms(0)

# Trains the online standard-scaler model.
model = standard_scaler.fit(input_data)

# Use the standard-scaler model for predictions.
output = model.transform(input_data)[0]

# extract and display the results
output = output.select(col("input"), col("output"), col("version"))
field_names = output.get_schema().get_field_names()

for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(standard_scaler.get_input_col())]
    output_value = result[field_names.index(standard_scaler.get_output_col())]
    model_version = result[field_names.index(standard_scaler.get_model_version_col())]
    print('Input Value: ' + str(input_value) + ' \tOutput Value: ' + str(output_value) +
          '\tModel Version: ' + str(model_version))
