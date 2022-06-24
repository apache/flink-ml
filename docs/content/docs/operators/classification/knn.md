---
title: "KNN"
type: docs
aliases:
- /operators/classification/knn.html
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

# KNN

K Nearest Neighbor(KNN) is a classification algorithm. The basic assumption of
KNN is that if most of the nearest K neighbors of the provided sample belongs to
the same label, then it is highly probabl that the provided sample also belongs
to that label.

## Input Columns

| Param name  | Type    | Default      | Description      |
| :---------- | :------ | :----------- | :--------------- |
| featuresCol | Vector  | `"features"` | Feature vector   |
| labelCol    | Integer | `"label"`    | Label to predict |

## Output Columns

| Param name    | Type    | Default        | Description     |
| :------------ | :------ | :------------- | :-------------- |
| predictionCol | Integer | `"prediction"` | Predicted label |

## Parameters

Below are parameters required by `KnnModel`.

| Key           | Default        | Type    | Required | Description                      |
| ------------- | -------------- | ------- | -------- | -------------------------------- |
| K             | `5`            | Integer | no       | The number of nearest neighbors. |
| featuresCol   | `"features"`   | String  | no       | Features column name.            |
| predictionCol | `"prediction"` | String  | no       | Prediction column name.          |

`Knn` needs parameters above and also below.

| Key      | Default   | Type   | Required | Description        |
| -------- | --------- | ------ | -------- | ------------------ |
| labelCol | `"label"` | String | no       | Label column name. |

## Examples

{{< tabs knn >}}

{{< tab "Java">}}
```java
import org.apache.flink.ml.classification.knn.Knn;
import org.apache.flink.ml.classification.knn.KnnModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;

List<Row> trainRows =
  new ArrayList<>(
  Arrays.asList(
    Row.of(Vectors.dense(2.0, 3.0), 1.0),
    Row.of(Vectors.dense(2.1, 3.1), 1.0),
    Row.of(Vectors.dense(200.1, 300.1), 2.0),
    Row.of(Vectors.dense(200.2, 300.2), 2.0),
    Row.of(Vectors.dense(200.3, 300.3), 2.0),
    Row.of(Vectors.dense(200.4, 300.4), 2.0),
    Row.of(Vectors.dense(200.4, 300.4), 2.0),
    Row.of(Vectors.dense(200.6, 300.6), 2.0),
    Row.of(Vectors.dense(2.1, 3.1), 1.0),
    Row.of(Vectors.dense(2.1, 3.1), 1.0),
    Row.of(Vectors.dense(2.1, 3.1), 1.0),
    Row.of(Vectors.dense(2.1, 3.1), 1.0),
    Row.of(Vectors.dense(2.3, 3.2), 1.0),
    Row.of(Vectors.dense(2.3, 3.2), 1.0),
    Row.of(Vectors.dense(2.8, 3.2), 3.0),
    Row.of(Vectors.dense(300., 3.2), 4.0),
    Row.of(Vectors.dense(2.2, 3.2), 1.0),
    Row.of(Vectors.dense(2.4, 3.2), 5.0),
    Row.of(Vectors.dense(2.5, 3.2), 5.0),
    Row.of(Vectors.dense(2.5, 3.2), 5.0),
    Row.of(Vectors.dense(2.1, 3.1), 1.0)));
List<Row> predictRows =
  new ArrayList<>(
  Arrays.asList(
    Row.of(Vectors.dense(4.0, 4.1), 5.0),
    Row.of(Vectors.dense(300, 42), 2.0)));
Schema schema =
  Schema.newBuilder()
  .column("f0", DataTypes.of(DenseVector.class))
  .column("f1", DataTypes.DOUBLE())
  .build();

DataStream<Row> dataStream = env.fromCollection(trainRows);
Table trainData = tEnv.fromDataStream(dataStream, schema).as("features", "label");
DataStream<Row> predDataStream = env.fromCollection(predictRows);
Table predictData = tEnv.fromDataStream(predDataStream, schema).as("features", "label");

Knn knn = new Knn();
KnnModel knnModel = knn.fit(trainData);
Table output = knnModel.transform(predictData)[0];

output.execute().print();
```
{{< /tab>}}

{{< tab "Python">}}
```python
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.classification.knn import KNN

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# load flink ml jar
env.add_jars("file:///{path}/statefun-flink-core-3.1.0.jar", "file:///{path}/flink-ml-uber-{version}.jar")

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

train_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([2.0, 3.0]), 1.0),
        (Vectors.dense([2.1, 3.1]), 1.0),
        (Vectors.dense([200.1, 300.1]), 2.0),
        (Vectors.dense([200.2, 300.2]), 2.0),
        (Vectors.dense([200.3, 300.3]), 2.0),
        (Vectors.dense([200.4, 300.4]), 2.0),
        (Vectors.dense([200.4, 300.4]), 2.0),
        (Vectors.dense([200.6, 300.6]), 2.0),
        (Vectors.dense([2.1, 3.1]), 1.0),
        (Vectors.dense([2.1, 3.1]), 1.0),
        (Vectors.dense([2.1, 3.1]), 1.0),
        (Vectors.dense([2.1, 3.1]), 1.0),
        (Vectors.dense([2.3, 3.2]), 1.0),
        (Vectors.dense([2.3, 3.2]), 1.0),
        (Vectors.dense([2.8, 3.2]), 3.0),
        (Vectors.dense([300., 3.2]), 4.0),
        (Vectors.dense([2.2, 3.2]), 1.0),
        (Vectors.dense([2.4, 3.2]), 5.0),
        (Vectors.dense([2.5, 3.2]), 5.0),
        (Vectors.dense([2.5, 3.2]), 5.0),
        (Vectors.dense([2.1, 3.1]), 1.0)
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label'],
            [DenseVectorTypeInfo(), Types.DOUBLE()])))

predict_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([4.0, 4.1]), 5.0),
        (Vectors.dense([300, 42]), 2.0),
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label'],
            [DenseVectorTypeInfo(), Types.DOUBLE()])))

knn = KNN()
model = knn.fit(train_data)
output = model.transform(predict_data)[0]
output.execute().print()

# output
# +----+--------------------------------+--------------------------------+--------------------------------+
# | op |                       features |                          label |                     prediction |
# +----+--------------------------------+--------------------------------+--------------------------------+
# | +I |                     [4.0, 4.1] |                            5.0 |                            5.0 |
# | +I |                  [300.0, 42.0] |                            2.0 |                            2.0 |
# +----+--------------------------------+--------------------------------+--------------------------------+
```
{{< /tab>}}
{{< /tabs>}}

