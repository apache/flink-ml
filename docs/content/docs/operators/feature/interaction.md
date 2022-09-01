---
title: "Interaction"
weight: 1
type: docs
aliases:
- /operators/feature/interaction.html
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

## Interaction

Interaction takes vector or numerical columns, and generates a single vector column that contains
the product of all combinations of one value from each input column.

For example, when the input feature values are Double(2) and Vector(3, 4), the output would be 
Vector(6, 8). When the input feature values are Vector(1, 2) and Vector(3, 4), the output would
be Vector(3, 4, 6, 8). If you change the position of these two input Vectors, the output would 
be Vector(3, 6, 4, 8).

### Input Columns

| Param name | Type   | Default | Description               |
|:-----------|:-------|:--------|:--------------------------|
| inputCols  | Vector | `null`  | Columns to be interacted. |

### Output Columns

| Param name | Type   | Default    | Description        |
|:-----------|:-------|:-----------|:-------------------|
| outputCol  | Vector | `"output"` | Interacted vector. |

### Parameters

| Key             | Default    | Type      | Required | Description                |
|-----------------|------------|-----------|----------|----------------------------|
| inputCols       | `null`     | String[]  | yes      | Input column names.        |
| outputCol       | `"output"` | String    | no       | Output column name.        |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.interaction.Interaction;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates an Interaction instance and uses it for feature engineering. */
public class InteractionExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(0, Vectors.dense(1.1, 3.2), Vectors.dense(2, 3)),
                        Row.of(1, Vectors.dense(2.1, 3.1), Vectors.dense(1, 3)));

        Table inputTable = tEnv.fromDataStream(inputStream).as("f0", "f1", "f2");

        // Creates an Interaction object and initializes its parameters.
        Interaction interaction =
                new Interaction().setInputCols("f0", "f1", "f2").setOutputCol("outputVec");

        // Transforms input data.
        Table outputTable = interaction.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            Object[] inputValues = new Object[interaction.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = row.getField(interaction.getInputCols()[i]);
            }
            Vector outputValue = (Vector) row.getField(interaction.getOutputCol());
            System.out.printf(
                    "Input Values: %s \tOutput Value: %s\n",
                    Arrays.toString(inputValues), outputValue);
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates an Interaction instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.feature.interaction import Interaction
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (1,
         Vectors.dense(1, 2),
         Vectors.dense(3, 4)),
        (2,
         Vectors.dense(2, 8),
         Vectors.dense(3, 4))
    ],
        type_info=Types.ROW_NAMED(
            ['f0', 'f1', 'f2'],
            [Types.INT(), DenseVectorTypeInfo(), DenseVectorTypeInfo()])))

# create an interaction object and initialize its parameters
interaction = Interaction() \
    .set_input_cols('f0', 'f1', 'f2') \
    .set_output_col('interaction_vec')

# use the interaction for feature engineering
output = interaction.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in interaction.get_input_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(interaction.get_input_cols())):
        input_values[i] = result[field_names.index(interaction.get_input_cols()[i])]
    output_value = result[field_names.index(interaction.get_output_col())]
    print('Input Values: ' + str(input_values) + '\tOutput Value: ' + str(output_value))

```

{{< /tab>}}

{{< /tabs>}}
