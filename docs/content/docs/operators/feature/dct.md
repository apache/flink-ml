---
title: "DCT"
weight: 1
type: docs
aliases:
- /operators/feature/dct.html
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

## DCT

DCT is a Transformer that takes the 1D discrete cosine transform of a real
vector. No zero padding is performed on the input vector. It returns a real
vector of the same length representing the DCT. The return vector is scaled such
that the transform matrix is unitary (aka scaled DCT-II).

### Input Columns

| Param name | Type   | Default   | Description                            |
|:-----------|:-------|:----------|:---------------------------------------|
| inputCol   | Vector | `"input"` | Input vector to be cosine transformed. |

### Output Columns

| Param name | Type   | Default    | Description                       |
|:-----------|:-------|:-----------|:----------------------------------|
| outputCol  | Vector | `"output"` | Cosine transformed output vector. |

### Parameters

| Key       | Default    | Type    | Required | Description                                                       |
|-----------|------------|---------|----------|-------------------------------------------------------------------|
| inputCol  | `"input"`  | String  | no       | Input column name.                                                |
| outputCol | `"output"` | String  | no       | Output column name.                                               |
| inverse   | `false`    | Boolean | no       | Whether to perform the inverse DCT (true) or forward DCT (false). |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.dct.DCT;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.List;

/** Simple program that creates a DCT instance and uses it for feature engineering. */
public class DCTExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        List<Vector> inputData =
                Arrays.asList(
                        Vectors.dense(1.0, 1.0, 1.0, 1.0), Vectors.dense(1.0, 0.0, -1.0, 0.0));
        Table inputTable = tEnv.fromDataStream(env.fromCollection(inputData)).as("input");

        // Creates a DCT object and initializes its parameters.
        DCT dct = new DCT();

        // Uses the DCT object for feature transformations.
        Table outputTable = dct.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            Vector inputValue = row.getFieldAs(dct.getInputCol());
            Vector outputValue = row.getFieldAs(dct.getOutputCol());

            System.out.printf("Input Value: %s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}
```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a DCT instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.dct import DCT
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(1.0, 1.0, 1.0, 1.0),),
        (Vectors.dense(1.0, 0.0, -1.0, 0.0),),
    ],
        type_info=Types.ROW_NAMED(
            ['input'],
            [DenseVectorTypeInfo()])))

# create a DCT object and initialize its parameters
dct = DCT()

# use the dct for feature engineering
output = dct.transform(input_data)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(dct.get_input_col())]
    output_value = result[field_names.index(dct.get_output_col())]
    print('Input Value: ' + str(input_value) + '\tOutput Value: ' + str(output_value))
```

{{< /tab>}}

{{< /tabs>}}
