---
title: "One Hot Encoder"
weight: 1
type: docs
aliases:
- /operators/feature/onehotencoder.html
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

# One Hot Encoder

One-hot encoding maps a categorical feature, represented as a label index, to a
binary vector with at most a single one-value indicating the presence of a
specific feature value from among the set of all feature values. This encoding
allows algorithms which expect continuous features, such as Logistic Regression,
to use categorical features.

OneHotEncoder can transform multiple columns, returning an one-hot-encoded
output vector column for each input column.

## Input Columns

| Param name | Type    | Default | Description |
| :--------- | :------ | :------ | :---------- |
| inputCols  | Integer | `null`  | Label index |

## Output Columns

| Param name | Type   | Default | Description           |
| :--------- | :----- | :------ | :-------------------- |
| outputCols | Vector | `null`  | Encoded binary vector |

## Parameters

| Key           | Default                          | Type    | Required | Description                                                  |
| ------------- | -------------------------------- | ------- | -------- | ------------------------------------------------------------ |
| inputCols     | `null`                           | String  | yes      | Input column names.                                          |
| outputCols    | `null`                           | String  | yes      | Output column names.                                         |
| handleInvalid | `HasHandleInvalid.ERROR_INVALID` | String  | No       | Strategy to handle invalid entries. Supported values: `HasHandleInvalid.ERROR_INVALID`, `HasHandleInvalid.SKIP_INVALID` |
| dropLast      | `true`                           | Boolean | no       | Whether to drop the last category.                           |

## Examples

```java
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoder;
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoderModel;

List<Row> trainData = Arrays.asList(Row.of(0.0), Row.of(1.0), Row.of(2.0), Row.of(0.0));
Table trainTable = tEnv.fromDataStream(env.fromCollection(trainData)).as("input");

List<Row> predictData = Arrays.asList(Row.of(0.0), Row.of(1.0), Row.of(2.0));
Table predictTable = tEnv.fromDataStream(env.fromCollection(predictData)).as("input");

OneHotEncoder estimator = new OneHotEncoder().setInputCols("input").setOutputCols("output");
OneHotEncoderModel model = estimator.fit(trainTable);
Table outputTable = model.transform(predictTable)[0];

outputTable.execute().print();
```







