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

# Simple program that trains a KMeans model and uses it for clustering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.clustering.kmeans import KMeans
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([0.0, 0.0]),),
        (Vectors.dense([0.0, 0.3]),),
        (Vectors.dense([0.3, 3.0]),),
        (Vectors.dense([9.0, 0.0]),),
        (Vectors.dense([9.0, 0.6]),),
        (Vectors.dense([9.6, 0.0]),),
    ],
        type_info=Types.ROW_NAMED(
            ['features'],
            [DenseVectorTypeInfo()])))

# create a kmeans object and initialize its parameters
kmeans = KMeans().set_k(2).set_seed(1)

# train the kmeans model
model = kmeans.fit(input_data)

# use the kmeans model for predictions
output = model.transform(input_data)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    features = result[field_names.index(kmeans.get_features_col())]
    cluster_id = result[field_names.index(kmeans.get_prediction_col())]
    print('Features: ' + str(features) + ' \tCluster Id: ' + str(cluster_id))
