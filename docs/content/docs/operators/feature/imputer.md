---
title: "Imputer"
weight: 1
type: docs
aliases:
- /operators/feature/imputer.html
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
specific language governing permissions dand limitations
under the License.
-->

## Imputer
The imputer for completing missing values of the input columns.

Missing values can be imputed using the statistics(mean, median or 
most frequent) of each column in which the missing values are located.
The input columns should be of numeric type.

__Note__ The `mean`/`median`/`most frequent` value is computed after 
filtering out missing values and null values, null values are always 
treated as missing, and so are also imputed.

__Note__ The parameter `relativeError` is only effective when the strategy
 is `median`.

### Input Columns

| Param name | Type   | Default | Description             |
|:-----------|:-------|:--------|:------------------------|
| inputCols  | Number | `null`  | Features to be imputed. |

### Output Columns

| Param name | Type   | Default | Description       |
|:-----------|:-------|:--------|:------------------|
| outputCols | Double | `null`  | Imputed features. |

### Parameters

Below are the parameters required by `ImputerModel`.

| Key           | Default      | Type        | Required | Description                                                                                |
|:--------------|:-------------|:------------|:---------|:-------------------------------------------------------------------------------------------|
| inputCols     | `null`       | String[]    | yes      | Input column names.                                                                        |
| outputCols    | `null`       | String[]    | yes      | Output column names.                                                                       |
| missingValue  | `Double.NaN` | Double      | no       | The placeholder for the missing values. All occurrences of missing values will be imputed. |

`Imputer` needs parameters above and also below.

| Key           | Default      | Type        | Required | Description                                                                   |
|:--------------|:-------------|:------------|:---------|:------------------------------------------------------------------------------|
| strategy      | `"mean"`     | String      | no       | The imputation strategy. Supported values: 'mean', 'median', 'most_frequent'. |
| relativeError | `0.001`      | Double      | no       | The relative target precision for the approximate quantile algorithm.         |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.imputer.Imputer;
import org.apache.flink.ml.feature.imputer.ImputerModel;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that trains a {@link Imputer} model and uses it for feature engineering. */
public class ImputerExample {

    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of(Double.NaN, 9.0),
                        Row.of(1.0, 9.0),
                        Row.of(1.5, 9.0),
                        Row.of(2.5, Double.NaN),
                        Row.of(5.0, 5.0),
                        Row.of(5.0, 4.0));
        Table trainTable = tEnv.fromDataStream(trainStream).as("input1", "input2");

        // Creates an Imputer object and initialize its parameters
        Imputer imputer =
                new Imputer()
                        .setInputCols("input1", "input2")
                        .setOutputCols("output1", "output2")
                        .setStrategy("mean")
                        .setMissingValue(Double.NaN);

        // Trains the Imputer model.
        ImputerModel model = imputer.fit(trainTable);

        // Uses the Imputer model for predictions.
        Table outputTable = model.transform(trainTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            double[] inputValues = new double[imputer.getInputCols().length];
            double[] outputValues = new double[imputer.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = (double) row.getField(imputer.getInputCols()[i]);
                outputValues[i] = (double) row.getField(imputer.getOutputCols()[i]);
            }
            System.out.printf(
                    "Input Values: %s\tOutput Values: %s\n",
                    Arrays.toString(inputValues), Arrays.toString(outputValues));
        }
    }
}
```

{{< /tab>}}

{{< tab "Python">}}

```python

# Simple program that creates an Imputer instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.lib.feature.imputer import Imputer
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input training and prediction data
train_data = t_env.from_data_stream(
    env.from_collection([
        (float('NaN'), 9.0,),
        (1.0, 9.0,),
        (1.5, 7.0,),
        (1.5, float('NaN'),),
        (4.0, 5.0,),
        (None, 4.0,),
    ],
        type_info=Types.ROW_NAMED(
            ['input1', 'input2'],
            [Types.DOUBLE(), Types.DOUBLE()])
    ))

# Creates an Imputer object and initializes its parameters.
imputer = Imputer()\
    .set_input_cols('input1', 'input2')\
    .set_output_cols('output1', 'output2')\
    .set_strategy('mean')\
    .set_missing_value(float('NaN'))

# Trains the Imputer Model.
model = imputer.fit(train_data)

# Uses the Imputer Model for predictions.
output = model.transform(train_data)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_values = []
    output_values = []
    for i in range(len(imputer.get_input_cols())):
        input_values.append(result[field_names.index(imputer.get_input_cols()[i])])
        output_values.append(result[field_names.index(imputer.get_output_cols()[i])])
    print('Input Values: ' + str(input_values) + '\tOutput Values: ' + str(output_values))
```

{{< /tab>}}

{{< /tabs>}}
