---
title: "Binarizer"
weight: 1
type: docs
aliases:
- /operators/feature/binarizer.html
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

## Binarizer

Binarizer binarizes the columns of continuous features by the given thresholds.
The continuous features may be DenseVector, SparseVector, or Numerical Value.

### Input Columns

| Param name | Type          | Default | Description                     |
|:-----------|:--------------|:--------|:--------------------------------|
| inputCols  | Number/Vector | `null`  | Number/Vectors to be binarized. |

### Output Columns

| Param name | Type          | Default | Description               |
|:-----------|:--------------|:--------|:--------------------------|
| outputCols | Number/Vector | `null`  | Binarized Number/Vectors. |

### Parameters

| Key         | Default   | Type     | Required | Description                                          |
|-------------|-----------|----------|----------|------------------------------------------------------|
| inputCols   | `null`    | String[] | yes      | Input column names.                                  |
| outputCols  | `null`    | String[] | yes      | Output column name.                                  |
| thresholds  | `null`    | Double[] | yes      | The thresholds used to binarize continuous features. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.binarizer.Binarizer;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates a Binarizer instance and uses it for feature engineering. */
public class BinarizerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(
                                1,
                                Vectors.dense(1, 2),
                                Vectors.sparse(
                                        17, new int[] {0, 3, 9}, new double[] {1.0, 2.0, 7.0})),
                        Row.of(
                                2,
                                Vectors.dense(2, 1),
                                Vectors.sparse(
                                        17, new int[] {0, 2, 14}, new double[] {5.0, 4.0, 1.0})),
                        Row.of(
                                3,
                                Vectors.dense(5, 18),
                                Vectors.sparse(
                                        17, new int[] {0, 11, 12}, new double[] {2.0, 4.0, 4.0})));

        Table inputTable = tEnv.fromDataStream(inputStream).as("f0", "f1", "f2");

        // Creates a Binarizer object and initializes its parameters.
        Binarizer binarizer =
                new Binarizer()
                        .setInputCols("f0", "f1", "f2")
                        .setOutputCols("of0", "of1", "of2")
                        .setThresholds(0.0, 0.0, 0.0);

        // Transforms input data.
        Table outputTable = binarizer.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            Object[] inputValues = new Object[binarizer.getInputCols().length];
            Object[] outputValues = new Object[binarizer.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = row.getField(binarizer.getInputCols()[i]);
                outputValues[i] = row.getField(binarizer.getOutputCols()[i]);
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
# Simple program that creates a Binarizer instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.feature.binarizer import Binarizer
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (1,
         Vectors.dense(3, 4)),
        (2,
         Vectors.dense(6, 2))
    ],
        type_info=Types.ROW_NAMED(
            ['f0', 'f1'],
            [Types.INT(), DenseVectorTypeInfo()])))

# create an binarizer object and initialize its parameters
binarizer = Binarizer() \
    .set_input_cols('f0', 'f1') \
    .set_output_cols('of0', 'of1') \
    .set_thresholds(1.5, 3.5)

# use the binarizer for feature engineering
output = binarizer.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in binarizer.get_input_cols()]
output_values = [None for _ in binarizer.get_output_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(binarizer.get_input_cols())):
        input_values[i] = result[field_names.index(binarizer.get_input_cols()[i])]
        output_values[i] = result[field_names.index(binarizer.get_output_cols()[i])]
    print('Input Values: ' + str(input_values) + '\tOutput Values: ' + str(output_values))

```

{{< /tab>}}

{{< /tabs>}}
