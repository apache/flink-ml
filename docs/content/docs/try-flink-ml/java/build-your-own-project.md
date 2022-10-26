---
title: "Building your own Flink ML project"
weight: 2
type: docs
aliases:
- /try-flink-ml/java/building-your-own-project.html
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

# Building your own Flink ML project

This document provides a quick introduction to using Flink ML. Readers of this
document will be guided to create a simple Flink job that trains a Machine
Learning Model and uses it to provide prediction service.

## What Will You Be Building?

Kmeans is a widely-used clustering algorithm and has been supported by Flink ML.
This walkthrough guides you to create a Flink job with Flink ML that initializes
and trains a Kmeans model, and finally uses it to predict the cluster id of
certain data points.

## Prerequisites

This walkthrough assumes that you have some familiarity with Java, but you
should be able to follow along even if you are coming from a different
programming language.

## Help, I’m Stuck!

If you get stuck, check out the [community support
resources](https://flink.apache.org/gettinghelp.html). In particular, Apache
Flink's [user mailing
list](https://flink.apache.org/community.html#mailing-lists) is consistently
ranked as one of the most active of any Apache project and a great way to get
help quickly.

## How To Follow Along

If you want to follow along, you will require a computer with:

- Java 8
- Maven 3

{{< unstable >}}

Before walking through the following sections of this document, make sure you
have downloaded Flink ML's latest code and installed Flink ML's Java SDK in your
local machine. You can refer to this [guideline]({{< ref
"docs/development/build-and-install#build-and-install-java-sdk" >}}) for how to
build and install Flink ML.

{{< /unstable >}}

While commands to be executed in a CLI are provided to walk through this example
in the following steps, it is recommended to use an IDE, like IntelliJ IDEA, to
manage, build and execute the example codes below.

Please use the following command to create a Flink Maven Archetype that provides
the basic skeleton of a project, with some necessary Flink dependencies.

```shell
$ mvn archetype:generate \
    -DarchetypeGroupId=org.apache.flink \
    -DarchetypeArtifactId=flink-quickstart-java \
    -DarchetypeVersion=1.15.1 \
    -DgroupId=kmeans-example \
    -DartifactId=kmeans-example \
    -Dversion=0.1 \
    -Dpackage=myflinkml \
    -DinteractiveMode=false
```

The command above would create a maven project named `kmeans-example` in your
current directory with the following structure:

```
$ tree kmeans-example
kmeans-example
├── pom.xml
└── src
    └── main
        ├── java
        │   └── myflinkml
        │       └── DataStreamJob.java
        └── resources
            └── log4j2.properties
```

Change the dependencies provided in `pom.xml` as follows:

```xml
<dependency>
  <groupId>org.apache.flink</groupId>
  <artifactId>flink-ml-uber</artifactId>
  <version>2.2-SNAPSHOT</version>
  <scope>provided</scope>
</dependency>

<dependency>
  <groupId>org.apache.flink</groupId>
  <artifactId>flink-connector-files</artifactId>
  <version>${flink.version}</version>
  <scope>provided</scope>
</dependency>

<dependency>
  <groupId>org.apache.flink</groupId>
  <artifactId>flink-clients</artifactId>
  <version>${flink.version}</version>
  <scope>provided</scope>
</dependency>

<dependency>
  <groupId>org.apache.flink</groupId>
  <artifactId>flink-table-api-java-bridge</artifactId>
  <version>${flink.version}</version>
  <scope>provided</scope>
</dependency>

<dependency>
  <groupId>org.apache.flink</groupId>
  <artifactId>flink-table-runtime</artifactId>
  <version>${flink.version}</version>
  <scope>provided</scope>
</dependency>		

<dependency>
  <groupId>org.apache.flink</groupId>
  <artifactId>flink-table-planner-loader</artifactId>
  <version>${flink.version}</version>
  <scope>provided</scope>
</dependency>

<dependency>
  <groupId>org.apache.flink</groupId>
  <artifactId>statefun-flink-core</artifactId>
  <version>3.2.0</version>
  <exclusions>
    <exclusion>
      <groupId>org.apache.flink</groupId>
      <artifactId>flink-streaming-java_2.12</artifactId>
    </exclusion>
    <exclusion>
      <groupId>org.apache.flink</groupId>
      <artifactId>flink-metrics-dropwizard</artifactId>
    </exclusion>
  </exclusions>
</dependency>
```

Create file `src/main/java/myflinkml/KMeansExample.java`, and save the following
content into the file. You may feel free to ignore and delete
`src/main/java/myflinkml/DataStreamJob.java` as it is not used in this
walkthrough.

```java
package myflinkml;

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

public class KMeansExample {
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

After placing the code above into your Maven project, you may use the following
command or your IDE to build and execute the example job.

```shell
cd kmeans-example/
mvn clean package
mvn exec:java -Dexec.mainClass="myflinkml.KMeansExample" -Dexec.classpathScope="compile"
```

If you are running the project in an IDE, you may get a
`java.lang.NoClassDefFoundError` exception. This is probably because you do not
have all required Flink dependencies implicitly loaded into the classpath.

- IntelliJ IDEA: Go to Run > Edit Configurations > Modify options > Select
  `include dependencies with "Provided" scope`. This run configuration will now
  include all required classes to run the application from within the IDE.

After executing the job, information like below will be printed out to your
terminal window.

```
Vector: [0.3, 0.0]	Cluster ID: 1
Vector: [9.6, 0.0]	Cluster ID: 0
Vector: [9.0, 0.6]	Cluster ID: 0
Vector: [0.0, 0.0]	Cluster ID: 1
Vector: [0.0, 0.3]	Cluster ID: 1
Vector: [9.0, 0.0]	Cluster ID: 0
```

<!-- TODO: figure out why the process above does not terminate. -->
The program might get stuck after printing out the information above, and you
may need to enter ^C to terminate the process. This only happens when the
program is executed locally and would not happen when the job is submitted to a
Flink cluster.

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
prediction process of the following Kmeans algorithm. Flink ML operators search
the names of the columns of the input table for input data, and produce
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

