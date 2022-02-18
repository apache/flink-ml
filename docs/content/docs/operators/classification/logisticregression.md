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

