---
title: "NGram"
weight: 1
type: docs
aliases:
- /operators/feature/ngram.html
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

## NGram
NGram converts the input string array into an array of n-grams, 
where each n-gram is represented by a space-separated string of
words. If the length of the input array is less than `n`, no 
n-grams are returned.

### Input Columns

| Param name | Type     | Default   | Description         |
|:-----------|:---------|:----------|:--------------------|
| inputCol   | String[] | `"input"` | Input string array. |

### Output Columns

| Param name | Type     | Default    | Description |
|:-----------|:---------|:-----------|:------------|
| outputCol  | String[] | `"output"` | N-grams.    |

### Parameters

| Key       | Default    | Type    | Required | Description                          |
|:----------|:-----------|:--------|:---------|:-------------------------------------|
| n         | `2`        | Integer | no       | Number of elements per n-gram (>=1). |
| inputCol  | `"input"`  | String  | no       | Input column name.                   |
| outputCol | `"output"` | String  | no       | Output column name.                  |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.ngram.NGram;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates an NGram instance and uses it for feature engineering. */
public class NGramExample {
	public static void main(String[] args) {
		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

		// Generates input data.
		DataStream<Row> inputStream =
			env.fromElements(
				Row.of((Object) new String[0]),
				Row.of((Object) new String[] {"a", "b", "c"}),
				Row.of((Object) new String[] {"a", "b", "c", "d"}));
		Table inputTable = tEnv.fromDataStream(inputStream).as("input");

		// Creates an NGram object and initializes its parameters.
		NGram nGram = new NGram().setN(2).setInputCol("input").setOutputCol("output");

		// Uses the NGram object for feature transformations.
		Table outputTable = nGram.transform(inputTable)[0];

		// Extracts and displays the results.
		for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
			Row row = it.next();

			String[] inputValue = (String[]) row.getField(nGram.getInputCol());
			String[] outputValue = (String[]) row.getField(nGram.getOutputCol());

			System.out.printf(
				"Input Value: %s \tOutput Value: %s\n",
				Arrays.toString(inputValue), Arrays.toString(outputValue));
		}
	}
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates an NGram instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.lib.feature.ngram import NGram
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()

t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_data_table = t_env.from_data_stream(
    env.from_collection([
        ([],),
        (['a', 'b', 'c'],),
        (['a', 'b', 'c', 'd'],),
    ],
        type_info=Types.ROW_NAMED(
            ["input", ],
            [Types.OBJECT_ARRAY(Types.STRING())])))

# Creates an NGram object and initializes its parameters.
n_gram = NGram() \
    .set_input_col('input') \
    .set_n(2) \
    .set_output_col('output')

# Uses the NGram object for feature transformations.
output = n_gram.transform(input_data_table)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(n_gram.get_input_col())]
    output_value = result[field_names.index(n_gram.get_output_col())]
    print('Input Value: ' + ' '.join(input_value) + '\tOutput Value: ' + str(output_value))

```

{{< /tab>}}

{{< /tabs>}}
