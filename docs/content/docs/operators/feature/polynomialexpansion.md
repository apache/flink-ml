---
title: "PolynomialExpansion"
weight: 1
type: docs
aliases:
- /operators/feature/polynomialexpansion.html
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

## PolynomialExpansion

A Transformer that expands the input vectors in polynomial space.

Take a 2-dimension vector as an example: `(x, y)`, if we want to expand it with degree 2, then
we get `(x, x * x, y, x * y, y * y)`.

<p>For more information about the polynomial expansion, see 
http://en.wikipedia.org/wiki/Polynomial_expansion.

### Input Columns

| Param name | Type   | Default   | Description             |
|:-----------|:-------|:----------|:------------------------|
| inputCol   | Vector | `"input"` | Vectors to be expanded. |

### Output Columns

| Param name | Type   | Default    | Description       |
|:-----------|:-------|:-----------|:------------------|
| outputCol  | Vector | `"output"` | Expanded vectors. |

### Parameters

| Key       | Default    | Type    | Required | Description                         |
|:----------|:-----------|:--------|:---------|:------------------------------------|
| inputCol  | `"input"`  | String  | no       | Input column name.                  |
| outputCol | `"output"` | String  | no       | Output column name.                 |
| degree    | `2`        | Integer | no       | Degree of the polynomial expansion. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.polynomialexpansion.PolynomialExpansion;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that creates a PolynomialExpansion instance and uses it for feature engineering. */
public class PolynomialExpansionExample {
	public static void main(String[] args) {
		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

		// Generates input data.
		DataStream<Row> inputStream =
			env.fromElements(
				Row.of(Vectors.dense(2.1, 3.1, 1.2)),
				Row.of(Vectors.dense(1.2, 3.1, 4.6)));
		Table inputTable = tEnv.fromDataStream(inputStream).as("inputVec");

		// Creates a PolynomialExpansion object and initializes its parameters.
		PolynomialExpansion polynomialExpansion =
			new PolynomialExpansion().setInputCol("inputVec").setDegree(2).setOutputCol("outputVec");

		// Uses the PolynomialExpansion object for feature transformations.
		Table outputTable = polynomialExpansion.transform(inputTable)[0];

		// Extracts and displays the results.
		for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
			Row row = it.next();

			Vector inputValue = (Vector) row.getField(polynomialExpansion.getInputCol());

			Vector outputValue = (Vector) row.getField(polynomialExpansion.getOutputCol());

			System.out.printf("Input Value: %s \tOutput Value: %s\n", inputValue, outputValue);
		}
	}
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a PolynomialExpansion instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.polynomialexpansion import PolynomialExpansion
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (1, Vectors.dense(2.1, 3.1, 1.2, 2.1)),
        (2, Vectors.dense(2.3, 2.1, 1.3, 1.2)),
    ],
        type_info=Types.ROW_NAMED(
            ['id', 'input_vec'],
            [Types.INT(), DenseVectorTypeInfo()])))

# create a polynomial expansion object and initialize its parameters
polynomialExpansion = PolynomialExpansion() \
    .set_input_col('input_vec') \
    .set_degree(2) \
    .set_output_col('output_vec')

# use the polynomial expansion model for feature engineering
output = polynomialExpansion.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(polynomialExpansion.get_input_col())]
    output_value = result[field_names.index(polynomialExpansion.get_output_col())]
    print('Input Value: ' + str(input_value) + '\tOutput Value: ' + str(output_value))

```

{{< /tab>}}

{{< /tabs>}}
