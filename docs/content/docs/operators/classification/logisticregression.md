---
title: "Logistic Regression"
type: docs
aliases:
- /operators/classification/logisticregression.html
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

# Logistic Regression

Logistic regression is a special case of Generalized Linear Model. It is widely
used to predict a binary response. 

## Input Columns

| Param name  | Type    | Default      | Description      |
| :---------- | :------ | :----------- | :--------------- |
| featuresCol | Vector  | `"features"` | Feature vector   |
| labelCol    | Integer | `"label"`    | Label to predict |
| weightCol   | Double  | `"weight"`   | Weight of sample |

## Output Columns

| Param name       | Type    | Default           | Description                             |
| :--------------- | :------ | :---------------- | :-------------------------------------- |
| predictionCol    | Integer | `"prediction"`    | Label of the max probability            |
| rawPredictionCol | Vector  | `"rawPrediction"` | Vector of the probability of each label |

## Parameters

Below are parameters required by `LogisticRegressionModel`.

| Key              | Default           | Type   | Required | Description                 |
| ---------------- | ----------------- | ------ | -------- | --------------------------- |
| featuresCol      | `"features"`      | String | no       | Features column name.       |
| predictionCol    | `"prediction"`    | String | no       | Prediction column name.     |
| rawPredictionCol | `"rawPrediction"` | String | no       | Raw prediction column name. |

`LogisticRegression` needs parameters above and also below.

| Key             | Default   | Type    | Required | Description                                                  |
| --------------- | --------- | ------- | -------- | ------------------------------------------------------------ |
| labelCol        | `"label"` | String  | no       | Label column name.                                           |
| weightCol       | `null`    | String  | no       | Weight column name.                                          |
| maxIter         | `20`      | Integer | no       | Maximum number of iterations.                                |
| reg             | `0.`      | Double  | no       | Regularization parameter.                                    |
| learningRate    | `0.1`     | Double  | no       | Learning rate of optimization method.                        |
| globalBatchSize | `32`      | Integer | no       | Global batch size of training algorithms.                    |
| tol             | `1e-6`    | Double  | no       | Convergence tolerance for iterative algorithms.              |
| multiClass      | `"auto"`  | String  | no       | Classification type. Supported values: "auto", "binomial", "multinomial" |

## Examples
{{< tabs logisticregression >}}

{{< tab "Java">}}
```java
import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;

List<Row> binomialTrainData =
  Arrays.asList(
  Row.of(Vectors.dense(1, 2, 3, 4), 0., 1.),
  Row.of(Vectors.dense(2, 2, 3, 4), 0., 2.),
  Row.of(Vectors.dense(3, 2, 3, 4), 0., 3.),
  Row.of(Vectors.dense(4, 2, 3, 4), 0., 4.),
  Row.of(Vectors.dense(5, 2, 3, 4), 0., 5.),
  Row.of(Vectors.dense(11, 2, 3, 4), 1., 1.),
  Row.of(Vectors.dense(12, 2, 3, 4), 1., 2.),
  Row.of(Vectors.dense(13, 2, 3, 4), 1., 3.),
  Row.of(Vectors.dense(14, 2, 3, 4), 1., 4.),
  Row.of(Vectors.dense(15, 2, 3, 4), 1., 5.));
Collections.shuffle(binomialTrainData);

Table binomialDataTable =
  tEnv.fromDataStream(
  env.fromCollection(
    binomialTrainData,
    new RowTypeInfo(
      new TypeInformation[] {
        TypeInformation.of(DenseVector.class),
        Types.DOUBLE,
        Types.DOUBLE
      },
      new String[] {"features", "label", "weight"})));

LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
LogisticRegressionModel model = logisticRegression.fit(binomialDataTable);
Table output = model.transform(binomialDataTable)[0];

output.execute().print();
```
{{< /tab>}}

{{< tab "Python">}}
```python
from pyflink.common import Types
from pyflink.table import StreamTableEnvironment

from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.classification.logisticregression import LogisticRegression

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# load flink ml jar
env.add_jars("file:///{path}/statefun-flink-core-3.1.0.jar;file:///{path}/flink-ml-uber-{version}.jar")

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

binomial_data_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense([1, 2, 3, 4]), 0., 1.),
        (Vectors.dense([2, 2, 3, 4]), 0., 2.),
        (Vectors.dense([3, 2, 3, 4]), 0., 3.),
        (Vectors.dense([4, 2, 3, 4]), 0., 4.),
        (Vectors.dense([5, 2, 3, 4]), 0., 5.),
        (Vectors.dense([11, 2, 3, 4]), 1., 1.),
        (Vectors.dense([12, 2, 3, 4]), 1., 2.),
        (Vectors.dense([13, 2, 3, 4]), 1., 3.),
        (Vectors.dense([14, 2, 3, 4]), 1., 4.),
        (Vectors.dense([15, 2, 3, 4]), 1., 5.),
    ],
        type_info=Types.ROW_NAMED(
            ['features', 'label', 'weight'],
            [DenseVectorTypeInfo(), Types.DOUBLE(), Types.DOUBLE()])
    ))

logistic_regression = LogisticRegression().set_weight_col('weight')
model = logistic_regression.fit(binomial_data_table)
output = model.transform(binomial_data_table)[0]

output.execute().print()

# output
# +----+--------------------------------+--------------------------------+--------------------------------+--------------------------------+--------------------------------+
# | op |                       features |                          label |                         weight |                     prediction |                  rawPrediction |
# +----+--------------------------------+--------------------------------+--------------------------------+--------------------------------+--------------------------------+
# | +I |           [1.0, 2.0, 3.0, 4.0] |                            0.0 |                            1.0 |                            0.0 | [0.9731815427669942, 0.0268... |
# | +I |           [5.0, 2.0, 3.0, 4.0] |                            0.0 |                            5.0 |                            0.0 | [0.8158018538556746, 0.1841... |
# | +I |          [14.0, 2.0, 3.0, 4.0] |                            1.0 |                            4.0 |                            1.0 | [0.03753179912156068, 0.962... |
# | +I |           [3.0, 2.0, 3.0, 4.0] |                            0.0 |                            3.0 |                            0.0 | [0.926886620226911, 0.07311... |
# | +I |          [12.0, 2.0, 3.0, 4.0] |                            1.0 |                            2.0 |                            1.0 | [0.10041228069167174, 0.899... |
# | +I |           [4.0, 2.0, 3.0, 4.0] |                            0.0 |                            4.0 |                            0.0 | [0.8822580948141717, 0.1177... |
# | +I |          [13.0, 2.0, 3.0, 4.0] |                            1.0 |                            3.0 |                            1.0 | [0.061891528893188164, 0.93... |
# | +I |           [2.0, 2.0, 3.0, 4.0] |                            0.0 |                            2.0 |                            0.0 | [0.9554533965544176, 0.0445... |
# | +I |          [11.0, 2.0, 3.0, 4.0] |                            1.0 |                            1.0 |                            1.0 | [0.15884837044317868, 0.841... |
# | +I |          [15.0, 2.0, 3.0, 4.0] |                            1.0 |                            5.0 |                            1.0 | [0.022529496926532833, 0.97... |
# +----+--------------------------------+--------------------------------+--------------------------------+--------------------------------+--------------------------------+
```
{{< /tab>}}
{{< /tabs>}}
