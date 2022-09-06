---
title: "Normalizer"
weight: 1
type: docs
aliases:
- /operators/feature/normalizer.html
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

## Normalizer

A Transformer that normalizes a vector to have unit norm using the given p-norm.

### Input Columns

| Param name | Type   | Default   | Description               |
|:-----------|:-------|:----------|:--------------------------|
| inputCol   | Vector | `"input"` | Vectors to be normalized. |

### Output Columns

| Param name | Type   | Default    | Description         |
|:-----------|:-------|:-----------|:--------------------|
| outputCol  | Vector | `"output"` | Normalized vectors. |

### Parameters

| Key       | Default    | Type   | Required | Description         |
|:----------|:-----------|:-------|:---------|:--------------------|
| inputCol  | `"input"`  | String | no       | Input column name.  |
| outputCol | `"output"` | String | no       | Output column name. |
| p         | `2.0`      | Double | no       | The p norm value.   |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.normalizer.Normalizer;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that creates a Normalizer instance and uses it for feature engineering. */
public class NormalizerExample {
	public static void main(String[] args) {
		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

		// Generates input data.
		DataStream<Row> inputStream =
			env.fromElements(
				Row.of(Vectors.dense(2.1, 3.1, 1.2, 3.1, 4.6)),
				Row.of(Vectors.dense(1.2, 3.1, 4.6, 2.1, 3.1)));
		Table inputTable = tEnv.fromDataStream(inputStream).as("inputVec");

		// Creates a Normalizer object and initializes its parameters.
		Normalizer normalizer =
			new Normalizer().setInputCol("inputVec").setP(3.0).setOutputCol("outputVec");

		// Uses the Normalizer object for feature transformations.
		Table outputTable = normalizer.transform(inputTable)[0];

		// Extracts and displays the results.
		for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
			Row row = it.next();

			Vector inputValue = (Vector) row.getField(normalizer.getInputCol());

			Vector outputValue = (Vector) row.getField(normalizer.getOutputCol());

			System.out.printf("Input Value: %s \tOutput Value: %s\n", inputValue, outputValue);
		}
	}
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a Normalizer instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.feature.normalizer import Normalizer
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

# create a normalizer object and initialize its parameters
normalizer = Normalizer() \
    .set_input_col('input_vec') \
    .set_p(1.5) \
    .set_output_col('output_vec')

# use the normalizer model for feature engineering
output = normalizer.transform(input_data_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(normalizer.get_input_col())]
    output_value = result[field_names.index(normalizer.get_output_col())]
    print('Input Value: ' + str(input_value) + '\tOutput Value: ' + str(output_value))

```

{{< /tab>}}

{{< /tabs>}}
