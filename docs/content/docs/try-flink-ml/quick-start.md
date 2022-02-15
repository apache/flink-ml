---
title: "Quick Start"
weight: 1
type: docs
aliases:
- /try-flink-ml/quick-start.html
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

# Quick Start

This document provides a quick introduction to using Flink ML. Readers of this
document will be guided to create a simple Flink job that trains a Machine
Learning Model and use it to provide prediction service.

## Maven Setup

In order to use Flink ML in a Maven project, add the following dependencies to
`pom.xml`.

{{< artifact flink-ml-core withScalaVersion >}}

{{< artifact flink-ml-iteration withScalaVersion >}}

{{< artifact flink-ml-lib withScalaVersion >}}

The example code provided in this document requires additional dependencies on
the Flink Table API. In order to execute the example code successfully, please
make sure the following dependencies also exist in `pom.xml`.

```xml
<dependency>
  <groupId>org.apache.flink</groupId>
  <artifactId>flink-clients_2.12</artifactId>
  <version>1.14.0</version>
</dependency>
```

```xml
<dependency>
  <groupId>org.apache.flink</groupId>
  <artifactId>flink-table-planner_2.12</artifactId>
  <version>1.14.0</version>
</dependency>
```

## Flink ML Example

Kmeans is a widely-used clustering algorithm and has been supported by Flink ML.
The example code below creates a Flink job with Flink ML that initializes and
trains a Kmeans model, and finally uses it to predict the cluster id of certain
data points.

```java
import org.apache.flink.ml.clustering.kmeans.KMeans;
import org.apache.flink.ml.clustering.kmeans.KMeansModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

public class QuickStart {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        String featuresCol = "features";
        String predictionCol = "prediction";

        // Generate train data and predict data as DataStream.
        DataStream<DenseVector> inputStream = env.fromElements(
                Vectors.dense(0.0, 0.0),
                Vectors.dense(0.0, 0.3),
                Vectors.dense(0.3, 0.0),
                Vectors.dense(9.0, 0.0),
                Vectors.dense(9.0, 0.6),
                Vectors.dense(9.6, 0.0)
        );

        // Convert data from DataStream to Table, as Flink ML uses Table API.
        Table input = tEnv.fromDataStream(inputStream).as(featuresCol);

        // Creates a K-means object and initialize its parameters.
        KMeans kmeans = new KMeans()
                .setK(2)
                .setSeed(1L)
                .setFeaturesCol(featuresCol)
                .setPredictionCol(predictionCol);

        // Trains the K-means Model.
        KMeansModel model = kmeans.fit(input);

        // Use the K-means Model for predictions.
        Table output = model.transform(input)[0];

        // Extracts and displays prediction result.
        for (CloseableIterator<Row> it = output.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector vector = (DenseVector) row.getField(featuresCol);
            int clusterId = (Integer) row.getField(predictionCol);
            System.out.println("Vector: " + vector + "\tCluster ID: " + clusterId);
        }
    }
}
```

After placing the code above into your Maven project and executing it,
information like below will be printed out to your terminal window.

```
Vector: [0.3, 0.0]	Cluster ID: 1
Vector: [9.6, 0.0]	Cluster ID: 0
Vector: [9.0, 0.6]	Cluster ID: 0
Vector: [0.0, 0.0]	Cluster ID: 1
Vector: [0.0, 0.3]	Cluster ID: 1
Vector: [9.0, 0.0]	Cluster ID: 0
```

## Breaking Down The Code

### The Execution Environment

The first lines set up the `StreamExecutionEnvironment` to execute the Flink ML
job. You would have been familiar with this concept if you have experience using
Flink. For the example program in this document, a simple
`StreamExecutionEnvironment` without specific configurations would be enough. 

Given that Flink ML uses Flink's Table API, a `StreamTableEnvironment` would
also be necessary for the following program.

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
```

### Creating Training & Inference Data Table

Then the program creates the Table containing data for the training and
prediction process of the following Kmeans algorithm. Flink ML operators
search the names of the columns of the input table for input data, and produce
prediction results to designated column of the output Table.

```java
DataStream<DenseVector> inputStream = env.fromElements(
        Vectors.dense(0.0, 0.0),
        Vectors.dense(0.0, 0.3),
        Vectors.dense(0.3, 0.0),
        Vectors.dense(9.0, 0.0),
        Vectors.dense(9.0, 0.6),
        Vectors.dense(9.6, 0.0)
);

Table input = tEnv.fromDataStream(inputStream).as(featuresCol);
```

### Creating, Configuring, Training & Using Kmeans

Flink ML classes for Kmeans algorithm include `KMeans` and `KMeansModel`.
`KMeans` implements the training process of Kmeans algorithm based on the
provided training data, and finally generates a `KMeansModel`.
`KmeansModel.transform()` method encodes the Transformation logic of this
algorithm and is used for predictions. 

Both `KMeans` and `KMeansModel` provides getter/setter methods for Kmeans
algorithm's configuration parameters. The example program explicitly sets the
following parameters, and other configuration parameters will have their default
values used.

- `K`, the number of clusters to create
- `seed`, the random seed to initialize cluster centers
- `featuresCol`, name of the column containing input feature vectors
- `predictionCol`, name of the column to output prediction results

When the program invokes `KMeans.fit()` to generate a `KMeansModel`, the
`KMeansModel` will inherit the `KMeans` object's configuration parameters. Thus
it is supported to set `KMeansModel`'s parameters directly in `KMeans` object.

```java
KMeans kmeans = new KMeans()
        .setK(2)
        .setSeed(1L)
        .setFeaturesCol(featuresCol)
        .setPredictionCol(predictionCol);

KMeansModel model = kmeans.fit(input);

Table output = model.transform(input)[0];

```

### Collecting Prediction Result

Like all other Flink programs, the codes described in the sections above only
configures the computation graph of a Flink job, and the program only evaluates
the computation logic and collects outputs after the `execute()` method is
invoked. Collected outputs from the output table would be `Row`s in which
`featuresCol` contains input feature vectors, and `predictionCol` contains
output prediction results, i.e., cluster IDs.

```java
for (CloseableIterator<Row> it = output.execute().collect(); it.hasNext(); ) {
    Row row = it.next();
    DenseVector vector = (DenseVector) row.getField(featuresCol);
    int clusterId = (Integer) row.getField(predictionCol);
    System.out.println("Vector: " + vector + "\tCluster ID: " + clusterId);
}
```

```
Vector: [0.3, 0.0]	Cluster ID: 1
Vector: [9.6, 0.0]	Cluster ID: 0
Vector: [9.0, 0.6]	Cluster ID: 0
Vector: [0.0, 0.0]	Cluster ID: 1
Vector: [0.0, 0.3]	Cluster ID: 1
Vector: [9.0, 0.0]	Cluster ID: 0
```

<!-- TODO: Add sections like "Next Steps` with guidance to other pages of this document. -->
