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

# Simple program that creates a Bucketizer instance and uses it for feature
# engineering.
#
# Before executing this program, please make sure you have followed Flink ML's
# quick start guideline to set up Flink ML and Flink environment. The guideline
# can be found at
#
# https://nightlies.apache.org/flink/flink-ml-docs-master/docs/try-flink-ml/quick-start/

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.clustering.agglomerativeclustering import AgglomerativeClustering
from pyflink.table import StreamTableEnvironment

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([1, 1]),),
        (Vectors.dense([1, 4]),),
        (Vectors.dense([1, 0]),),
        (Vectors.dense([4, 1.5]),),
        (Vectors.dense([4, 4]),),
        (Vectors.dense([4, 0]),),
    ],
        type_info=Types.ROW_NAMED(
            ['features'],
            [DenseVectorTypeInfo()])))

# Creates an AgglomerativeClustering object and initializes its parameters.
agglomerative_clustering = AgglomerativeClustering() \
    .set_linkage('ward') \
    .set_distance_measure('euclidean') \
    .set_prediction_col('prediction') \
    .set_compute_full_tree(True)

# Uses the AgglomerativeClustering for clustering.
outputs = agglomerative_clustering.transform(input_data)

# Extracts and display the results.
field_names = outputs[0].get_schema().get_field_names()
for result in t_env.to_data_stream(outputs[0]).execute_and_collect():
    features = result[field_names.index(agglomerative_clustering.features_col)]
    cluster_id = result[field_names.index(agglomerative_clustering.prediction_col)]
    print('Features: ' + str(features) + '\tCluster ID: ' + str(cluster_id))

"""
# The following code snippet could be used to visualize the merge info.

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

merge_info = [result for result in
              t_env.to_data_stream(outputs[1]).execute_and_collect()]
plt.title("Agglomerative Clustering Dendrogram")
dendrogram(merge_info)
plt.xlabel("Index of data point.")
plt.ylabel("Distances between merged clusters.")
plt.show()
"""
