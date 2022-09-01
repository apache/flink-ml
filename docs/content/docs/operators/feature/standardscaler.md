---
title: "Standard Scaler"
weight: 1
type: docs
aliases:
- /operators/feature/standardscaler.html
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

## Standard Scaler

Standard Scaler is an algorithm that standardizes the input features by removing
the mean and scaling each dimension to unit variance.
### Input Columns

| Param name | Type   | Default   | Description            |
|:-----------|:-------|:----------|:-----------------------|
| inputCol   | Vector | `"input"` | Features to be scaled. |

### Output Columns

| Param name | Type   | Default    | Description      |
|:-----------|:-------|:-----------|:-----------------|
| outputCol  | Vector | `"output"` | Scaled features. |

### Parameters

| Key       | Default    | Type    | Required | Description                                        |
|-----------|------------|---------|----------|----------------------------------------------------|
| inputCol  | `"input"`  | String  | no       | Input column name.                                 |
| outputCol | `"output"` | String  | no       | Output column name.                                |
| withMean  | `false`    | Boolean | no       | Whether centers the data with mean before scaling. |
| withStd   | `true`     | Boolean | no       | Whether scales the data with standard deviation.   |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.standardscaler.StandardScaler;
import org.apache.flink.ml.feature.standardscaler.StandardScalerModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a StandardScaler model and uses it for feature engineering. */
public class StandardScalerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(Vectors.dense(-2.5, 9, 1)),
                        Row.of(Vectors.dense(1.4, -5, 1)),
                        Row.of(Vectors.dense(2, -1, -2)));
        Table inputTable = tEnv.fromDataStream(inputStream).as("input");

        // Creates a StandardScaler object and initializes its parameters.
        StandardScaler standardScaler = new StandardScaler();

        // Trains the StandardScaler Model.
        StandardScalerModel model = standardScaler.fit(inputTable);

        // Uses the StandardScaler Model for predictions.
        Table outputTable = model.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector inputValue = (DenseVector) row.getField(standardScaler.getInputCol());
            DenseVector outputValue = (DenseVector) row.getField(standardScaler.getOutputCol());
            System.out.printf("Input Value: %s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that trains a StandardScaler model and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.feature.standardscaler import StandardScaler
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(-2.5, 9, 1),),
        (Vectors.dense(1.4, -5, 1),),
        (Vectors.dense(2, -1, -2),),
    ],
        type_info=Types.ROW_NAMED(
            ['input'],
            [DenseVectorTypeInfo()])
    ))

# create a standard-scaler object and initialize its parameters
standard_scaler = StandardScaler()

# train the standard-scaler model
model = standard_scaler.fit(input_data)

# use the standard-scaler model for predictions
output = model.transform(input_data)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(standard_scaler.get_input_col())]
    output_value = result[field_names.index(standard_scaler.get_output_col())]
    print('Input Value: ' + str(input_value) + ' \tOutput Value: ' + str(output_value))

```

{{< /tab>}}

{{< /tabs>}}
