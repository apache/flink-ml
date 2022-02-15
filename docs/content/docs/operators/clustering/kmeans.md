---
title: "Kmeans"
weight: 1
type: docs
aliases:
- /operators/clustering/kmeans.html
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

# K-means

K-means is a commonly-used clustering algorithm. It groups given data points
into a predefined number of clusters.

## Input Columns

| Param name  | Type    | Default    | Description    |
| :---------- | :------ | :--------- | :------------- |
| featuresCol | Vector  | "features" | Feature vector |

## Output Columns

| Param name    | Type    | Default      | Description              |
| :------------ | :------ | :----------- | :----------------------- |
| predictionCol | Integer | "prediction" | Predicted cluster center |

## Parameters

Below are parameters required by `KmeansModel`.

| Key             | Default                         | Type   | Required | Description                                                  |
| --------------- | ------------------------------- | ------ | -------- | ------------------------------------------------------------ |
| distanceMeasure | `EuclideanDistanceMeasure.NAME` | String | no       | Distance measure. Supported values: `EuclideanDistanceMeasure.NAME` |
| featuresCol     | `"features"`                    | String | no       | Features column name.                                        |
| predictionCol   | `"prediction"`                  | String | no       | Prediction column name.                                      |

`Kmeans` need parameters above and also below.

| Key      | Default    | Type    | Required | Description                                                |
| -------- | ---------- | ------- | -------- | ---------------------------------------------------------- |
| initMode | `"random"` | String  | no       | The initialization algorithm. Supported options: 'random'. |
| seed     | `null`     | Long    | no       | The random seed.                                           |
| maxIter  | `20`       | Integer | no       | Maximum number of iterations.                              |

## Examples

```java
import org.apache.flink.ml.clustering.kmeans.KMeans;
import org.apache.flink.ml.clustering.kmeans.KMeansModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;

// Generates train data and predict data.
DataStream<DenseVector> inputStream = env.fromElements(
  Vectors.dense(0.0, 0.0),
  Vectors.dense(0.0, 0.3),
  Vectors.dense(0.3, 0.0),
  Vectors.dense(9.0, 0.0),
  Vectors.dense(9.0, 0.6),
  Vectors.dense(9.6, 0.0)
);
Table input = tEnv.fromDataStream(inputStream).as("features");

// Creates a K-means object and initialize its parameters.
KMeans kmeans = new KMeans()
  .setK(2)
  .setSeed(1L);

// Trains the K-means Model.
KMeansModel model = kmeans.fit(input);

// Uses the K-means Model to do predictions.
Table output = model.transform(input)[0];

// Extracts and displays prediction result.
for (CloseableIterator<Row> it = output.execute().collect(); it.hasNext(); ) {
  Row row = it.next();
  DenseVector vector = (DenseVector) row.getField("features");
  int clusterId = (Integer) row.getField("prediction");
  System.out.println("Vector: " + vector + "\tCluster ID: " + clusterId);
}
```
