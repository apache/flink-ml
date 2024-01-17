---
title: "OneHotEncoder"
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

## OneHotEncoder

OneHotEncoder maps a categorical feature, represented as a label index, to a
binary vector with at most a single one-value indicating the presence of a
specific feature value from among the set of all feature values. This encoding
allows algorithms that expect continuous features, such as Logistic Regression,
to use categorical features.

OneHotEncoder can transform multiple columns, returning a one-hot-encoded output
vector column for each input column.

### Input Columns

| Param name | Type    | Default | Description  |
| :--------- | :------ | :------ |:-------------|
| inputCols  | Integer | `null`  | Label index. |

### Output Columns

| Param name | Type   | Default | Description            |
| :--------- | :----- | :------ |:-----------------------|
| outputCols | Vector | `null`  | Encoded binary vector. |

### Parameters

| Key           | Default   | Type     | Required | Description                                                                    |
|---------------|-----------|----------|----------|--------------------------------------------------------------------------------|
| inputCols     | `null`    | String[] | yes      | Input column names.                                                            |
| outputCols    | `null`    | String[] | yes      | Output column names.                                                           |
| handleInvalid | `"error"` | String   | no       | Strategy to handle invalid entries. Supported values: 'error', 'skip', 'keep'. |
| dropLast      | `true`    | Boolean  | no       | Whether to drop the last category.                                             |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}
```java
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoder;
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoderModel;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a OneHotEncoder model and uses it for feature engineering. */
public class OneHotEncoderExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(Row.of(0.0), Row.of(1.0), Row.of(2.0), Row.of(0.0));
        Table trainTable = tEnv.fromDataStream(trainStream).as("input");

        DataStream<Row> predictStream = env.fromElements(Row.of(0.0), Row.of(1.0), Row.of(2.0));
        Table predictTable = tEnv.fromDataStream(predictStream).as("input");

        // Creates a OneHotEncoder object and initializes its parameters.
        OneHotEncoder oneHotEncoder =
                new OneHotEncoder().setInputCols("input").setOutputCols("output");

        // Trains the OneHotEncoder Model.
        OneHotEncoderModel model = oneHotEncoder.fit(trainTable);

        // Uses the OneHotEncoder Model for predictions.
        Table outputTable = model.transform(predictTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            Double inputValue = (Double) row.getField(oneHotEncoder.getInputCols()[0]);
            SparseVector outputValue =
                    (SparseVector) row.getField(oneHotEncoder.getOutputCols()[0]);
            System.out.printf("Input Value: %s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}

```
{{< /tab>}}

{{< tab "Python">}}
```python
# Simple program that trains a OneHotEncoder model and uses it for feature
# engineering.

from pyflink.common import Row
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.onehotencoder import OneHotEncoder
from pyflink.table import StreamTableEnvironment, DataTypes

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input training and prediction data
train_table = t_env.from_elements(
    [Row(0.0), Row(1.0), Row(2.0), Row(0.0)],
    DataTypes.ROW([
        DataTypes.FIELD('input', DataTypes.DOUBLE())
    ]))

predict_table = t_env.from_elements(
    [Row(0.0), Row(1.0), Row(2.0)],
    DataTypes.ROW([
        DataTypes.FIELD('input', DataTypes.DOUBLE())
    ]))

# create a one-hot-encoder object and initialize its parameters
one_hot_encoder = OneHotEncoder().set_input_cols('input').set_output_cols('output')

# train the one-hot-encoder model
model = one_hot_encoder.fit(train_table)

# use the one-hot-encoder model for predictions
output = model.transform(predict_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(one_hot_encoder.get_input_cols()[0])]
    output_value = result[field_names.index(one_hot_encoder.get_output_cols()[0])]
    print('Input Value: ' + str(input_value) + ' \tOutput Value: ' + str(output_value))

```
{{< /tab>}}

{{< /tabs>}}
