---
title: "Building your own Flink ML project"
weight: 2
type: docs
aliases:
- /try-flink-ml/python/building-your-own-project.html

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
Learning Model and use it to provide prediction service.

## Prerequisites

Python version (3.6, 3.7, or 3.8) is required for Flink ML. Please run the
following command to make sure that it meets the requirements:

```shell
$ python --version
# the version printed here must be 3.6, 3.7 or 3.8
```

## Installation of Flink ML Python SDK

Flink ML Python SDK is available in
[PyPi](https://pypi.org/project/apache-flink-ml/) and can be installed as
follows:

{{< stable >}}

```bash
$ python -m pip install apache-flink-ml=={{< version >}}
```

{{< /stable >}} {{< unstable >}}

```bash
$ python -m pip install apache-flink-ml
```

{{< /unstable >}}

You can also build Flink ML Python SDK from sources by following the
[development guide]({{< ref "docs/development/building" >}}).

## Flink ML Example

Kmeans is a widely-used clustering algorithm and has been supported by Flink ML.
The example code below creates a Flink job with Flink ML that initializes and
trains a Kmeans model, and finally uses it to predict the cluster id of certain
data points.

```python
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.clustering.kmeans import KMeans
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

After placing the code above into your Python file and executing it, information
like the below will be printed out to your terminal window.

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

```python
# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)
```

### Creating Training & Inference Data Table

Then the program creates the Table containing data for the training and
prediction process of the following Kmeans algorithm. Flink ML operators search
the names of the columns of the input table for input data, and produce
prediction results to designated column of the output Table.

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

### Collecting Prediction Result

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

```
Features: [9.6,0.0]     Cluster Id: 0
Features: [9.0,0.6]     Cluster Id: 0
Features: [0.0,0.3]     Cluster Id: 1
Features: [0.0,0.0]     Cluster Id: 1
Features: [0.3,3.0]     Cluster Id: 1
Features: [9.0,0.0]     Cluster Id: 0
```

