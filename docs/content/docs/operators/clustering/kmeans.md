---
title: "Kmeans"
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

## K-means

K-means is a commonly-used clustering algorithm. It groups given data points
into a predefined number of clusters.

### Input Columns

| Param name  | Type   | Default      | Description     |
|:------------|:-------|:-------------|:----------------|
| featuresCol | Vector | `"features"` | Feature vector. |

### Output Columns

| Param name    | Type    | Default        | Description               |
|:--------------|:--------|:---------------|:--------------------------|
| predictionCol | Integer | `"prediction"` | Predicted cluster center. |

### Parameters

Below are the parameters required by `KMeansModel`.

| Key             | Default        | Type    | Required | Description                                                               |
|-----------------|----------------|---------|----------|---------------------------------------------------------------------------|
| distanceMeasure | `euclidean`    | String  | no       | Distance measure. Supported values: `'euclidean', 'manhattan', 'cosine'`. |
| featuresCol     | `"features"`   | String  | no       | Features column name.                                                     |
| predictionCol   | `"prediction"` | String  | no       | Prediction column name.                                                   |
| k               | `2`            | Integer | no       | The max number of clusters to create.                                     |

`KMeans` needs parameters above and also below.

| Key      | Default    | Type    | Required | Description                                                |
|----------|------------|---------|----------|------------------------------------------------------------|
| initMode | `"random"` | String  | no       | The initialization algorithm. Supported options: 'random'. |
| seed     | `null`     | Long    | no       | The random seed.                                           |
| maxIter  | `20`       | Integer | no       | Maximum number of iterations.                              |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}
```java
import org.apache.flink.ml.clustering.kmeans.KMeans;
import org.apache.flink.ml.clustering.kmeans.KMeansModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a KMeans model and uses it for clustering. */
public class KMeansExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<DenseVector> inputStream =
                env.fromElements(
                        Vectors.dense(0.0, 0.0),
                        Vectors.dense(0.0, 0.3),
                        Vectors.dense(0.3, 0.0),
                        Vectors.dense(9.0, 0.0),
                        Vectors.dense(9.0, 0.6),
                        Vectors.dense(9.6, 0.0));
        Table inputTable = tEnv.fromDataStream(inputStream).as("features");

        // Creates a K-means object and initializes its parameters.
        KMeans kmeans = new KMeans().setK(2).setSeed(1L);

        // Trains the K-means Model.
        KMeansModel kmeansModel = kmeans.fit(inputTable);

        // Uses the K-means Model for predictions.
        Table outputTable = kmeansModel.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector features = (DenseVector) row.getField(kmeans.getFeaturesCol());
            int clusterId = (Integer) row.getField(kmeans.getPredictionCol());
            System.out.printf("Features: %s \tCluster ID: %s\n", features, clusterId);
        }
    }
}

```
{{< /tab>}}

{{< tab "Python">}}
```python
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

```
{{< /tab>}}

{{< /tabs>}}

## Online K-means

Online K-Means extends the function of K-Means, supporting to train a K-Means
model continuously according to an unbounded stream of train data. 

Online K-Means makes updates with the "mini-batch" K-Means rule, generalized to
incorporate forgetfulness (i.e. decay). After the centroids estimated on the
current batch are acquired, Online K-Means computes the new centroids from the
weighted average between the original and the estimated centroids. The weight of
the estimated centroids is the number of points assigned to them. The weight of
the original centroids is also the number of points, but additionally
multiplying with the decay factor. 

The decay factor scales the contribution of the clusters as estimated thus far.
If the decay factor is 1, all batches are weighted equally. If the decay factor
is 0, new centroids are determined entirely by recent data. Lower values
correspond to more forgetting.

### Input Columns

| Param name  | Type   | Default      | Description    |
|:------------|:-------|:-------------|:---------------|
| featuresCol | Vector | `"features"` | Feature vector |

### Output Columns

| Param name    | Type    | Default        | Description              |
|:--------------|:--------|:---------------|:-------------------------|
| predictionCol | Integer | `"prediction"` | Predicted cluster center |

### Parameters

Below are the parameters required by `OnlineKMeansModel`.

| Key             | Default        | Type    | Required | Description                                                               |
|-----------------|----------------|---------|----------|---------------------------------------------------------------------------|
| distanceMeasure | `euclidean`    | String  | no       | Distance measure. Supported values: `'euclidean', 'manhattan', 'cosine'`. |
| featuresCol     | `"features"`   | String  | no       | Features column name.                                                     |
| predictionCol   | `"prediction"` | String  | no       | Prediction column name.                                                   |
| k               | `2`            | Integer | no       | The max number of clusters to create.                                     |

`OnlineKMeans` needs parameters above and also below.

| Key             | Default          | Type    | Required | Description                                           |
|-----------------|------------------|---------|----------|-------------------------------------------------------|
| batchStrategy   | `COUNT_STRATEGY` | String  | no       | Strategy to create mini batch from online train data. |
| globalBatchSize | `32`             | Integer | no       | Global batch size of training algorithms.             |
| decayFactor     | `0.`             | Double  | no       | The forgetfulness of the previous centroids.          |
| seed            | null             | Long    | no       | The random seed.                                      |

### Examples

{{< tabs online_examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.clustering.kmeans.KMeansModelData;
import org.apache.flink.ml.clustering.kmeans.OnlineKMeans;
import org.apache.flink.ml.clustering.kmeans.OnlineKMeansModel;
import org.apache.flink.ml.examples.util.PeriodicSourceFunction;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Simple program that trains an OnlineKMeans model and uses it for clustering. */
public class OnlineKMeansExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data. Both are infinite streams that periodically
        // sends out provided data to trigger model update and prediction.
        List<Row> trainData1 =
                Arrays.asList(
                        Row.of(Vectors.dense(0.0, 0.0)),
                        Row.of(Vectors.dense(0.0, 0.3)),
                        Row.of(Vectors.dense(0.3, 0.0)),
                        Row.of(Vectors.dense(9.0, 0.0)),
                        Row.of(Vectors.dense(9.0, 0.6)),
                        Row.of(Vectors.dense(9.6, 0.0)));

        List<Row> trainData2 =
                Arrays.asList(
                        Row.of(Vectors.dense(10.0, 100.0)),
                        Row.of(Vectors.dense(10.0, 100.3)),
                        Row.of(Vectors.dense(10.3, 100.0)),
                        Row.of(Vectors.dense(-10.0, -100.0)),
                        Row.of(Vectors.dense(-10.0, -100.6)),
                        Row.of(Vectors.dense(-10.6, -100.0)));

        List<Row> predictData =
                Arrays.asList(
                        Row.of(Vectors.dense(10.0, 10.0)), Row.of(Vectors.dense(-10.0, 10.0)));

        SourceFunction<Row> trainSource =
                new PeriodicSourceFunction(1000, Arrays.asList(trainData1, trainData2));
        DataStream<Row> trainStream =
                env.addSource(trainSource, new RowTypeInfo(DenseVectorTypeInfo.INSTANCE));
        Table trainTable = tEnv.fromDataStream(trainStream).as("features");

        SourceFunction<Row> predictSource =
                new PeriodicSourceFunction(1000, Collections.singletonList(predictData));
        DataStream<Row> predictStream =
                env.addSource(predictSource, new RowTypeInfo(DenseVectorTypeInfo.INSTANCE));
        Table predictTable = tEnv.fromDataStream(predictStream).as("features");

        // Creates an online K-means object and initializes its parameters and initial model data.
        OnlineKMeans onlineKMeans =
                new OnlineKMeans()
                        .setFeaturesCol("features")
                        .setPredictionCol("prediction")
                        .setGlobalBatchSize(6)
                        .setInitialModelData(
                                KMeansModelData.generateRandomModelData(tEnv, 2, 2, 0.0, 0));

        // Trains the online K-means Model.
        OnlineKMeansModel onlineModel = onlineKMeans.fit(trainTable);

        // Uses the online K-means Model for predictions.
        Table outputTable = onlineModel.transform(predictTable)[0];

        // Extracts and displays the results. As training data stream continuously triggers the
        // update of the internal k-means model data, clustering results of the same predict dataset
        // would change over time.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row1 = it.next();
            DenseVector features1 = (DenseVector) row1.getField(onlineKMeans.getFeaturesCol());
            Integer clusterId1 = (Integer) row1.getField(onlineKMeans.getPredictionCol());
            Row row2 = it.next();
            DenseVector features2 = (DenseVector) row2.getField(onlineKMeans.getFeaturesCol());
            Integer clusterId2 = (Integer) row2.getField(onlineKMeans.getPredictionCol());
            if (Objects.equals(clusterId1, clusterId2)) {
                System.out.printf("%s and %s are now in the same cluster.\n", features1, features2);
            } else {
                System.out.printf(
                        "%s and %s are now in different clusters.\n", features1, features2);
            }
        }
    }
}

```

{{< /tab>}}

{{< /tabs>}}
