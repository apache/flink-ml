---
title: "Naive Bayes"
type: docs
aliases:
- /operators/classification/naivebayes.html
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

# Naive Bayes

Naive Bayes is a multiclass classifier. Based on Bayes’ theorem, it assumes that
there is strong (naive) independence between every pair of features. 

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

Below are parameters required by `NaiveBayesModel`.

| Key           | Default         | Type   | Required | Description                                     |
| ------------- | --------------- | ------ | -------- | ----------------------------------------------- |
| modelType     | `"multinomial"` | String | no       | The model type. Supported values: "multinomial" |
| featuresCol   | `"features"`    | String | no       | Features column name.                           |
| predictionCol | `"prediction"`  | String | no       | Prediction column name.                         |

`NaiveBayes` needs parameters above and also below.

| Key       | Default   | Type   | Required | Description              |
| --------- | --------- | ------ | -------- | ------------------------ |
| labelCol  | `"label"` | String | no       | Label column name.       |
| smoothing | `1.0`     | Double | no       | The smoothing parameter. |

## Examples

{{< tabs naivebayes >}}

{{< tab "Java">}}
```java
import org.apache.flink.ml.classification.naivebayes.NaiveBayes;
import org.apache.flink.ml.classification.naivebayes.NaiveBayesModel;
import org.apache.flink.ml.linalg.Vectors;

List<Row> trainData =
  Arrays.asList(
  Row.of(Vectors.dense(0, 0.), 11),
  Row.of(Vectors.dense(1, 0), 10),
  Row.of(Vectors.dense(1, 1.), 10));

Table trainTable = tEnv.fromDataStream(env.fromCollection(trainData)).as("features", "label");

List<Row> predictData =
  Arrays.asList(
  Row.of(Vectors.dense(0, 1.)),
  Row.of(Vectors.dense(0, 0.)),
  Row.of(Vectors.dense(1, 0)),
  Row.of(Vectors.dense(1, 1.)));

Table predictTable = tEnv.fromDataStream(env.fromCollection(predictData)).as("features");

NaiveBayes estimator =
  new NaiveBayes()
  .setSmoothing(1.0)
  .setFeaturesCol("features")
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setModelType("multinomial");

NaiveBayesModel model = estimator.fit(trainTable);
Table outputTable = model.transform(predictTable)[0];

outputTable.execute().print();
```
{{< /tab>}}


{{< tab "Python">}}
```python
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.classification.naivebayes import NaiveBayes

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# load flink ml jar
env.add_jars("file:///{path}/statefun-flink-core-3.1.0.jar", "file:///{path}/flink-ml-uber-{version}.jar")

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

train_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([0, 0.]), 11.),
        (Vectors.dense([1, 0]), 10.),
        (Vectors.dense([1, 1.]), 10.),
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label'],
            [DenseVectorTypeInfo(), Types.DOUBLE()])))

predict_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([0, 1.]),),
        (Vectors.dense([0, 0.]),),
        (Vectors.dense([1, 0]),),
        (Vectors.dense([1, 1.]),),
    ],
        type_info=Types.ROW_NAMED(
            ['features'],
            [DenseVectorTypeInfo()])))

estimator = (NaiveBayes()
             .set_smoothing(1.0)
             .set_features_col('features')
             .set_label_col('label')
             .set_prediction_col('prediction')
             .set_model_type('multinomial'))

model = estimator.fit(train_table)
output = model.transform(predict_table)[0]

print([result for result in t_env.to_data_stream(output).execute_and_collect()])

# output
# [<Row(DenseVector([0.0, 0.0]), 11.0)>, <Row(DenseVector([1.0, 0.0]), 10.0)>, <Row(DenseVector([0.0, 1.0]), 11.0)>, <Row(DenseVector([1.0, 1.0]), 10.0)>]

```
{{< /tab>}}
{{< /tabs>}}
