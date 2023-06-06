---
title: "HashingTF"
weight: 1
type: docs
aliases:
- /operators/feature/hashingtf.html
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

## HashingTF

HashingTF maps a sequence of terms(strings, numbers, booleans)
to a sparse vector with a specified dimension using the hashing
trick. If multiple features are projected into the same column,
the output values are accumulated by default.

### Input Columns

| Param name | Type                                          | Default   | Description              |
|:-----------|:----------------------------------------------|:----------|:-------------------------|
| inputCol   | List/Array of primitive data types or strings | `"input"` | Input sequence of terms. |

### Output Columns

| Param name | Type         | Default    | Description           |
|:-----------|:-------------|:-----------|:----------------------|
| outputCol  | SparseVector | `"output"` | Output sparse vector. |

### Parameters

| Key         | Default    | Type    | Required | Description                                                         |
|:------------|:-----------|:--------|:---------|:--------------------------------------------------------------------|
| binary      | `false`    | Boolean | no       | Whether each dimension of the output vector is binary or not.       |
| inputCol    | `"input"`  | String  | no       | Input column name.                                                  |
| outputCol   | `"output"` | String  | no       | Output column name.                                                 |
| numFeatures | `262144`   | Integer | no       | The number of features. It will be the length of the output vector. |


### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java

import org.apache.flink.ml.feature.hashingtf.HashingTF;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.List;

/** Simple program that creates a HashingTF instance and uses it for feature engineering. */
public class HashingTFExample {
	public static void main(String[] args) {
		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

		// Generates input data.
		DataStream<Row> inputStream =
			env.fromElements(
				Row.of(
					Arrays.asList(
						"HashingTFTest", "Hashing", "Term", "Frequency", "Test")),
				Row.of(
					Arrays.asList(
						"HashingTFTest", "Hashing", "Hashing", "Test", "Test")));

		Table inputTable = tEnv.fromDataStream(inputStream).as("input");

		// Creates a HashingTF object and initializes its parameters.
		HashingTF hashingTF =
			new HashingTF().setInputCol("input").setOutputCol("output").setNumFeatures(128);

		// Uses the HashingTF object for feature transformations.
		Table outputTable = hashingTF.transform(inputTable)[0];

		// Extracts and displays the results.
		for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
			Row row = it.next();

			List<Object> inputValue = (List<Object>) row.getField(hashingTF.getInputCol());
			SparseVector outputValue = (SparseVector) row.getField(hashingTF.getOutputCol());

			System.out.printf(
				"Input Value: %s \tOutput Value: %s\n",
				Arrays.toString(inputValue.stream().toArray()), outputValue);
		}
	}
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a HashingTF instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.hashingtf import HashingTF
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()

t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_data_table = t_env.from_data_stream(
    env.from_collection([
        (['HashingTFTest', 'Hashing', 'Term', 'Frequency', 'Test'],),
        (['HashingTFTest', 'Hashing', 'Hashing', 'Test', 'Test'],),
    ],
        type_info=Types.ROW_NAMED(
            ["input", ],
            [Types.OBJECT_ARRAY(Types.STRING())])))

# Creates a HashingTF object and initializes its parameters.
hashing_tf = HashingTF() \
    .set_input_col('input') \
    .set_num_features(128) \
    .set_output_col('output')

# Uses the HashingTF object for feature transformations.
output = hashing_tf.transform(input_data_table)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(hashing_tf.get_input_col())]
    output_value = result[field_names.index(hashing_tf.get_output_col())]
    print('Input Value: ' + ' '.join(input_value) + '\tOutput Value: ' + str(output_value))

```

{{< /tab>}}

{{< /tabs>}}
