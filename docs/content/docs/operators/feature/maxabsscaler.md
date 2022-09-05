---
title: "Max Abs Scaler"
weight: 1
type: docs
aliases:
- /operators/feature/maxabsscaler.html
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

## Max Abs Scaler

Max Abs Scaler is an algorithm rescales feature values to the range [-1, 1] 
by dividing through the largest maximum absolute value in each feature. 
It does not shift/center the data and thus does not destroy any sparsity.

### Input Columns

| Param name | Type   | Default   | Description            |
|:-----------|:-------|:----------|:-----------------------|
| inputCol   | Vector | `"input"` | Features to be scaled. |

### Output Columns

| Param name | Type   | Default    | Description      |
|:-----------|:-------|:-----------|:-----------------|
| outputCol  | Vector | `"output"` | Scaled features. |

### Parameters

| Key       | Default    | Type   | Required | Description         |
|-----------|------------|--------|----------|---------------------|
| inputCol  | `"input"`  | String | no       | Input column name.  |
| outputCol | `"output"` | String | no       | Output column name. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.maxabsscaler.MaxAbsScaler;
import org.apache.flink.ml.feature.maxabsscaler.MaxAbsScalerModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a MaxAbsScaler model and uses it for feature engineering. */
public class MaxAbsScalerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of(Vectors.dense(0.0, 3.0)),
                        Row.of(Vectors.dense(2.1, 0.0)),
                        Row.of(Vectors.dense(4.1, 5.1)),
                        Row.of(Vectors.dense(6.1, 8.1)),
                        Row.of(Vectors.dense(200, 400)));
        Table trainTable = tEnv.fromDataStream(trainStream).as("input");

        DataStream<Row> predictStream =
                env.fromElements(
                        Row.of(Vectors.dense(150.0, 90.0)),
                        Row.of(Vectors.dense(50.0, 40.0)),
                        Row.of(Vectors.dense(100.0, 50.0)));
        Table predictTable = tEnv.fromDataStream(predictStream).as("input");

        // Creates a MaxAbsScaler object and initializes its parameters.
        MaxAbsScaler maxAbsScaler = new MaxAbsScaler();

        // Trains the MaxAbsScaler Model.
        MaxAbsScalerModel maxAbsScalerModel = maxAbsScaler.fit(trainTable);

        // Uses the MaxAbsScaler Model for predictions.
        Table outputTable = maxAbsScalerModel.transform(predictTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector inputValue = (DenseVector) row.getField(maxAbsScaler.getInputCol());
            DenseVector outputValue = (DenseVector) row.getField(maxAbsScaler.getOutputCol());
            System.out.printf("Input Value: %-15s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that trains a MaxAbsScaler model and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.feature.maxabsscaler import MaxAbsScaler
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input training and prediction data
train_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(0.0, 3.0),),
        (Vectors.dense(2.1, 0.0),),
        (Vectors.dense(4.1, 5.1),),
        (Vectors.dense(6.1, 8.1),),
        (Vectors.dense(200, 400),),
    ],
        type_info=Types.ROW_NAMED(
            ['input'],
            [DenseVectorTypeInfo()])
    ))

predict_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(150.0, 90.0),),
        (Vectors.dense(50.0, 40.0),),
        (Vectors.dense(100.0, 50.0),),
    ],
        type_info=Types.ROW_NAMED(
            ['input'],
            [DenseVectorTypeInfo()])
    ))

# create a maxabs scaler object and initialize its parameters
max_abs_scaler = MaxAbsScaler()

# train the maxabs scaler model
model = max_abs_scaler.fit(train_data)

# use the maxabs scaler model for predictions
output = model.transform(predict_data)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(max_abs_scaler.get_input_col())]
    output_value = result[field_names.index(max_abs_scaler.get_output_col())]
    print('Input Value: ' + str(input_value) + ' \tOutput Value: ' + str(output_value))

```

{{< /tab>}}

{{< /tabs>}}
