---
title: "ElementwiseProduct"
weight: 1
type: docs
aliases:
- /operators/feature/elementwiseproduct.html
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

## ElementwiseProduct

ElementwiseProduct multiplies each input vector with a given scaling vector using 
Hadamard product. If the size of the input vector does not equal the size of the 
scaling vector, the transformer will throw an IllegalArgumentException.

### Input Columns

| Param name | Type   | Default   | Description            |
|:-----------|:-------|:----------|:-----------------------|
| inputCol   | Vector | `"input"` | Features to be scaled. |

### Output Columns

| Param name | Type   | Default    | Description      |
|:-----------|:-------|:-----------|:-----------------|
| outputCol  | Vector | `"output"` | Scaled features. |

### Parameters

| Key        | Default    | Type   | Required | Description         |
|------------|------------|--------|----------|---------------------|
| inputCol   | `"input"`  | String | no       | Input column name.  |
| outputCol  | `"output"` | String | no       | Output column name. |
| scalingVec | `null`     | String | yes      | The scaling vector. |
### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.elementwiseproduct.ElementwiseProduct;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/**
 * Simple program that creates an ElementwiseProduct instance and uses it for feature engineering.
 */
public class ElementwiseProductExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(0, Vectors.dense(1.1, 3.2)), Row.of(1, Vectors.dense(2.1, 3.1)));

        Table inputTable = tEnv.fromDataStream(inputStream).as("id", "vec");

        // Creates an ElementwiseProduct object and initializes its parameters.
        ElementwiseProduct elementwiseProduct =
                new ElementwiseProduct()
                        .setInputCol("vec")
                        .setOutputCol("outputVec")
                        .setScalingVec(Vectors.dense(1.1, 1.1));

        // Transforms input data.
        Table outputTable = elementwiseProduct.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            Vector inputValue = (Vector) row.getField(elementwiseProduct.getInputCol());
            Vector outputValue = (Vector) row.getField(elementwiseProduct.getOutputCol());
            System.out.printf("Input Value: %s \tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates an ElementwiseProduct instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.elementwiseproduct import ElementwiseProduct
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (1, Vectors.dense(2.1, 3.1)),
        (2, Vectors.dense(1.1, 3.3))
    ],
        type_info=Types.ROW_NAMED(
            ['id', 'vec'],
            [Types.INT(), DenseVectorTypeInfo()])))

# create an elementwise product object and initialize its parameters
elementwise_product = ElementwiseProduct() \
    .set_input_col('vec') \
    .set_output_col('output_vec') \
    .set_scaling_vec(Vectors.dense(1.1, 1.1))

# use the elementwise product object for feature engineering
output = elementwise_product.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(elementwise_product.get_input_col())]
    output_value = result[field_names.index(elementwise_product.get_output_col())]
    print('Input Value: ' + str(input_value) + '\tOutput Value: ' + str(output_value))

```

{{< /tab>}}

{{< /tabs>}}
