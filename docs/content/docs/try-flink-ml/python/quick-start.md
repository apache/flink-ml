---
title: "Quick Start"
weight: 1
type: docs
aliases:
- /try-flink-ml/python/quick-start.html

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
Learning Model and uses it to provide prediction service.

## What Will You Be Building?

Kmeans is a widely-used clustering algorithm and has been supported by Flink ML.
This walkthrough guides you to create a Flink job with Flink ML that initializes
and trains a Kmeans model, and finally uses it to predict the cluster id of
certain data points.

## Prerequisites

This walkthrough assumes that you have some familiarity with Python, but you
should be able to follow along even if you come from a different programming
language.

## Help, I’m Stuck!

If you get stuck, check out the [community support
resources](https://flink.apache.org/gettinghelp.html). In particular, Apache
Flink's [user mailing
list](https://flink.apache.org/community.html#mailing-lists) is consistently
ranked as one of the most active of any Apache project and a great way to get
help quickly.

## How To Follow Along

If you want to follow along, you will require a computer with:

{{< stable >}}
- Java 8
- Python 3.7 or 3.8 {{< /stable >}} {{< unstable >}}
- Java 8
- Maven 3
- Python 3.7 or 3.8 {{< /unstable >}}

{{< stable >}}

This walkthrough requires installing Flink ML Python SDK, which is available on
[PyPi](https://pypi.org/project/apache-flink-ml/) and can be easily installed
using pip.

```bash
$ python -m pip install apache-flink-ml=={{< version >}}
```

{{< /stable >}} {{< unstable >}}

Please walk through this [guideline]({{< ref
"docs/development/build-and-install#build-and-install-python-sdk" >}}) to build
and install Flink ML's Python SDK in your local environment.

{{< /unstable >}}

## Writing a Flink ML Python Program

Flink ML programs begin by setting up the `StreamExecutionEnvironment` to
execute the Flink ML job. You would have been familiar with this concept if you
have experience using Flink. For the example program in this document, a simple
`StreamExecutionEnvironment` without specific configurations would be enough. 

Given that Flink ML uses Flink's Table API, a `StreamTableEnvironment` would
also be necessary for the following program.

```python
# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)
```

Then you can create the Table containing data for the training and prediction
process of the following Kmeans algorithm. Flink ML operators search the names
of the columns of the input table for input data, and produce prediction results
to designated column of the output Table.

```python
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
```

Flink ML classes for Kmeans algorithm include `KMeans` and `KMeansModel`.
`KMeans` implements the training process of Kmeans algorithm based on the
provided training data, and finally generates a `KMeansModel`.
`KmeansModel.transform()` method encodes the Transformation logic of this
algorithm and is used for predictions. 

Both `KMeans` and `KMeansModel` provides getter/setter methods for Kmeans
algorithm's configuration parameters. This example program explicitly sets the
following parameters, and other configuration parameters will have their default
values used.

- `k`, the number of clusters to create
- `seed`, the random seed to initialize cluster centers

When the program invokes `KMeans.fit()` to generate a `KMeansModel`, the
`KMeansModel` will inherit the `KMeans` object's configuration parameters. Thus
it is supported to set `KMeansModel`'s parameters directly in `KMeans` object.

```python
# create a kmeans object and initialize its parameters
kmeans = KMeans().set_k(2).set_seed(1)

# train the kmeans model
model = kmeans.fit(input_data)

# use the kmeans model for predictions
output = model.transform(input_data)[0]

```

Like all other Flink programs, the codes described in the sections above only
configures the computation graph of a Flink job, and the program only evaluates
the computation logic and collects outputs after the `execute()` method is
invoked. Collected outputs from the output table would be `Row`s in which
`featuresCol` contains input feature vectors, and `predictionCol` contains
output prediction results, i.e., cluster IDs.

```python
# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    features = result[field_names.index(kmeans.get_features_col())]
    cluster_id = result[field_names.index(kmeans.get_prediction_col())]
    print('Features: ' + str(features) + ' \tCluster Id: ' + str(cluster_id))
```

The complete code so far:

```python
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

## Executing a Flink ML Python Program locally

After creating a python file (e.g. kmeans_example.py) and saving the code above
into the file, you can run the example on the command line:

```shell
python kmeans_example.py
```

The command above would build the example job and run it in a local mini
cluster. A sample output in your terminal is as follows.

```
Features: [9.6,0.0]     Cluster Id: 0
Features: [9.0,0.6]     Cluster Id: 0
Features: [0.0,0.3]     Cluster Id: 1
Features: [0.0,0.0]     Cluster Id: 1
Features: [0.3,3.0]     Cluster Id: 1
Features: [9.0,0.0]     Cluster Id: 0
```

## Executing a Flink ML Python Program on a Flink Cluster

### Prerequisites

Make sure Java 8 or a higher version has been installed in your local machine.
To check the Java version installed, type in your terminal:

```shell
$ java -version
```

### Download Flink

Download [Flink 1.17](https://flink.apache.org/downloads.html), then extract the archive:

```shell
$ tar -xzf flink-*.tgz
```

### Set Up Flink Library and Environment Variables

Run the following commands after having downloaded Flink:

```bash
cd ${path_to_flink}
cp opt/flink-python* lib/
export FLINK_HOME=`pwd`
```

### Add Flink ML library to Flink's library folder

You need to copy Flink ML's library files to Flink's folder for proper
initialization. 

{{< stable >}}

Please download [Flink ML Python
source](https://flink.apache.org/downloads.html) and extract the jar files into
Flink's library folder.

```shell
tar -xzf apache-flink-ml*.tar.gz
cp apache-flink-ml-*/deps/lib/* $FLINK_HOME/lib/
```

{{< /stable >}} {{< unstable >}}

Given that you have followed this [guideline]({{< ref
"docs/development/build-and-install#build-and-install-java-sdk" >}}), you
would have already built Flink ML's Java SDK. Now, you need to copy the
generated library files to Flink's folder with the following commands.

```shell
cd ${path_to_flink_ml}
cp ./flink-ml-dist/target/flink-ml-*-bin/flink-ml*/lib/*.jar $FLINK_HOME/lib/
```

{{< /unstable >}}

### Run Flink ML job

Please start a Flink standalone cluster in your local environment with the
following command.

```bash
$FLINK_HOME/bin/start-cluster.sh
```

You should be able to navigate to the web UI at
[localhost:8081](http://localhost:8081/) to view the Flink dashboard and see
that the cluster is up and running.


After creating a python file (e.g. kmeans_example.py) and saving the code above
into the file,  you may submit the example job to the cluster as follows.

```bash
$FLINK_HOME/bin/flink run -py kmeans_example.py
```

A sample output in your terminal is as follows.

```
Features: [9.6,0.0]     Cluster Id: 0
Features: [9.0,0.6]     Cluster Id: 0
Features: [0.0,0.3]     Cluster Id: 1
Features: [0.0,0.0]     Cluster Id: 1
Features: [0.3,3.0]     Cluster Id: 1
Features: [9.0,0.0]     Cluster Id: 0
```

Now you have successfully run the Flink ML job on a Flink Cluster. Other
detailed instructions to submit it to a Flink cluster can be found in [Job
Submission
Examples](https://nightlies.apache.org/flink/flink-docs-master/docs/deployment/cli/#submitting-pyflink-jobs).

Finally, you can stop the Flink standalone cluster with the following command.

```bash
$FLINK_HOME/bin/stop-cluster.sh
```
