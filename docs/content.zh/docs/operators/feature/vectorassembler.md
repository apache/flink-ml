---
title: "VectorAssembler"
weight: 1
type: docs
aliases:
- /operators/feature/vectorassembler.html
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

## VectorAssembler
A Transformer which combines a given list of input columns into a vector column. Input columns
would be numerical or vectors whose sizes are specified by the {@link #INPUT_SIZES} parameter.
Invalid input data with null values or values with wrong sizes would be dealt with according to
the strategy specified by the {@link HasHandleInvalid} parameter as follows:
<ul>
   <li>keep: If the input column data is null, a vector would be created with the specified size
       and NaN values. The vector would be used in the assembling process to represent the input
       column data. If the input column data is a vector, the data would be used in the assembling
       process even if it has a wrong size.
   <li>skip: If the input column data is null or a vector with wrong size, the input row would be
       filtered out and not be sent to downstream operators.
   <li>error: If the input column data is null or a vector with wrong size, an exception would be
       thrown.
</ul>
### Input Columns

| Param name | Type          | Default | Description                     |
|:-----------|:--------------|:--------|:--------------------------------|
| inputCols  | Number/Vector | `null`  | Number/Vectors to be assembled. |

### Output Columns

| Param name | Type   | Default    | Description       |
|:-----------|:-------|:-----------|:------------------|
| outputCol  | Vector | `"output"` | Assembled vector. |

### Parameters

| Key             | Default    | Type      | Required | Description                                                                    |
|-----------------|------------|-----------|----------|--------------------------------------------------------------------------------|
| inputCols       | `null`     | String[]  | yes      | Input column names.                                                            |
| outputCol       | `"output"` | String    | no       | Output column name.                                                            |
| inputSizes      | `null`     | Integer[] | yes      | Sizes of the input elements to be assembled.                                   |
| handleInvalid   | `"error"`  | String    | no       | Strategy to handle invalid entries. Supported values: 'error', 'skip', 'keep'. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.vectorassembler.VectorAssembler;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates a VectorAssembler instance and uses it for feature engineering. */
public class VectorAssemblerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(
                                Vectors.dense(2.1, 3.1),
                                1.0,
                                Vectors.sparse(5, new int[] {3}, new double[] {1.0})),
                        Row.of(
                                Vectors.dense(2.1, 3.1),
                                1.0,
                                Vectors.sparse(
                                        5,
                                        new int[] {4, 2, 3, 1},
                                        new double[] {4.0, 2.0, 3.0, 1.0})));
        Table inputTable = tEnv.fromDataStream(inputStream).as("vec", "num", "sparseVec");

        // Creates a VectorAssembler object and initializes its parameters.
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 5);

        // Uses the VectorAssembler object for feature transformations.
        Table outputTable = vectorAssembler.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            Object[] inputValues = new Object[vectorAssembler.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = row.getField(vectorAssembler.getInputCols()[i]);
            }

            Vector outputValue = (Vector) row.getField(vectorAssembler.getOutputCol());

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
# Simple program that creates a VectorAssembler instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo, SparseVectorTypeInfo
from pyflink.ml.feature.vectorassembler import VectorAssembler
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (Vectors.dense(2.1, 3.1),
         1.0,
         Vectors.sparse(5, [3], [1.0])),
        (Vectors.dense(2.1, 3.1),
         1.0,
         Vectors.sparse(5, [1, 2, 3, 4],
                        [1.0, 2.0, 3.0, 4.0])),
    ],
        type_info=Types.ROW_NAMED(
            ['vec', 'num', 'sparse_vec'],
            [DenseVectorTypeInfo(), Types.DOUBLE(), SparseVectorTypeInfo()])))

# create a vector assembler object and initialize its parameters
vector_assembler = VectorAssembler() \
    .set_input_cols('vec', 'num', 'sparse_vec') \
    .set_output_col('assembled_vec') \
    .set_input_sizes(2, 1, 5) \
    .set_handle_invalid('keep')

# use the vector assembler for feature engineering
output = vector_assembler.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in vector_assembler.get_input_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(vector_assembler.get_input_cols())):
        input_values[i] = result[field_names.index(vector_assembler.get_input_cols()[i])]
    output_value = result[field_names.index(vector_assembler.get_output_col())]
    print('Input Values: ' + str(input_values) + '\tOutput Value: ' + str(output_value))

```

{{< /tab>}}

{{< /tabs>}}
