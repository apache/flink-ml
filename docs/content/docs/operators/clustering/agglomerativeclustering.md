---
title: "AgglomerativeClustering"
type: docs
aliases:
- /operators/clustering/agglomerativeclustering.html
---
<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

## AgglomerativeClustering

AgglomerativeClustering performs a hierarchical clustering
using a bottom-up approach. Each observation starts in its 
own cluster and the clusters are merged together one by one.

The output contains two tables. The first one assigns one
cluster Id for each data point. The second one contains the
information of merging two clusters at each step. The data
format of the merging information is 
(clusterId1, clusterId2, distance, sizeOfMergedCluster).

### Input Columns

| Param name  | Type   | Default      | Description     |
|:------------|:-------|:-------------|:----------------|
| featuresCol | Vector | `"features"` | Feature vector. |

### Output Columns

| Param name    | Type    | Default        | Description               |
|:--------------|:--------|:---------------|:--------------------------|
| predictionCol | Integer | `"prediction"` | Predicted cluster center. |

### Parameters

| Key               | Default        | Type    | Required | Description                                                                                                         |
|:------------------|:---------------|:--------|:---------|:--------------------------------------------------------------------------------------------------------------------|
| numClusters       | `2`            | Integer | no       | The max number of clusters to create.                                                                               |
| distanceThreshold | `null`         | Double  | no       | Threshold to decide whether two clusters should be merged.                                                          |
| linkage           | `"ward"`       | String  | no       | Criterion for computing distance between two clusters. Supported values: `'ward', 'complete', 'single', 'average'`. |
| computeFullTree   | `false`        | Boolean | no       | Whether computes the full tree after convergence.                                                                   |
| distanceMeasure   | `"euclidean"`  | String  | no       | Distance measure. Supported values: `'euclidean', 'manhattan', 'cosine'`.                                           |
| featuresCol       | `"features"`   | String  | no       | Features column name.                                                                                               |
| predictionCol     | `"prediction"` | String  | no       | Prediction column name.                                                                                             |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}
```java
import org.apache.flink.ml.clustering.agglomerativeclustering.AgglomerativeClustering;
import org.apache.flink.ml.clustering.agglomerativeclustering.AgglomerativeClusteringParams;
import org.apache.flink.ml.common.distance.EuclideanDistanceMeasure;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that creates an AgglomerativeClustering instance and uses it for clustering. */
public class AgglomerativeClusteringExample {
	public static void main(String[] args) {
		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

		// Generates input data.
		DataStream<DenseVector> inputStream =
			env.fromElements(
				Vectors.dense(1, 1),
				Vectors.dense(1, 4),
				Vectors.dense(1, 0),
				Vectors.dense(4, 1.5),
				Vectors.dense(4, 4),
				Vectors.dense(4, 0));
		Table inputTable = tEnv.fromDataStream(inputStream).as("features");

		// Creates an AgglomerativeClustering object and initializes its parameters.
		AgglomerativeClustering agglomerativeClustering =
			new AgglomerativeClustering()
				.setLinkage(AgglomerativeClusteringParams.LINKAGE_WARD)
				.setDistanceMeasure(EuclideanDistanceMeasure.NAME)
				.setPredictionCol("prediction");

		// Uses the AgglomerativeClustering object for clustering.
		Table[] outputs = agglomerativeClustering.transform(inputTable);

		// Extracts and displays the results.
		for (CloseableIterator<Row> it = outputs[0].execute().collect(); it.hasNext(); ) {
			Row row = it.next();
			DenseVector features =
				(DenseVector) row.getField(agglomerativeClustering.getFeaturesCol());
			int clusterId = (Integer) row.getField(agglomerativeClustering.getPredictionCol());
			System.out.printf("Features: %s \tCluster ID: %s\n", features, clusterId);
		}
	}
}

```
{{< /tab>}}

{{< tab "Python">}}
```python
# Simple program that creates a Bucketizer instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.clustering.agglomerativeclustering import AgglomerativeClustering
from pyflink.table import StreamTableEnvironment
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

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
    .set_prediction_col('prediction')

# Uses the AgglomerativeClustering for clustering.
outputs = agglomerative_clustering.transform(input_data)

# Extracts and display the clustering results.
field_names = outputs[0].get_schema().get_field_names()
for result in t_env.to_data_stream(outputs[0]).execute_and_collect():
    features = result[field_names.index(agglomerative_clustering.features_col)]
    cluster_id = result[field_names.index(agglomerative_clustering.prediction_col)]
    print('Features: ' + str(features) + '\tCluster ID: ' + str(cluster_id))

# Visualizes the merge info.
merge_info = [result for result in
              t_env.to_data_stream(outputs[1]).execute_and_collect()]
plt.title("Agglomerative Clustering Dendrogram")
dendrogram(merge_info)
plt.xlabel("Index of data point.")
plt.ylabel("Distances between merged clusters.")
plt.show()
```
{{< /tab>}}

{{< /tabs>}}
