---
title: "RegexTokenizer"
weight: 1
type: docs
aliases:
- /operators/feature/regextokenizer.html
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

## RegexTokenizer

RegexTokenizer is an algorithm that converts the input string
to lowercase and then splits it by white spaces based on regex. 

### Input Columns

| Param name | Type   | Default   | Description              |
|:-----------|:-------|:----------|:-------------------------|
| inputCol   | String | `"input"` | Strings to be tokenized. |

### Output Columns

| Param name | Type     | Default    | Description        |
|:-----------|:---------|:-----------|:-------------------|
| outputCol  | String[] | `"output"` | Tokenized Strings. |

### Parameters

| Key            | Default    | Type    | Required | Description                                                       |
|:---------------|:-----------|:--------|:---------|:------------------------------------------------------------------|
| minTokenLength | `1`        | Integer | no       | Minimum token length.                                             |
| gaps           | `true`     | Boolean | no       | Set regex to match gaps or tokens.                                |
| pattern        | `"\s+"`    | String  | no       | Regex pattern used for tokenizing.                                |
| toLowercase    | `true`     | Boolean | no       | Whether to convert all characters to lowercase before tokenizing. |
| inputCol       | `"input"`  | String  | no       | Input column name.                                                |
| outputCol      | `"output"` | String  | no       | Output column name.                                               |
### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.regextokenizer.RegexTokenizer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates a RegexTokenizer instance and uses it for feature engineering. */
public class RegexTokenizerExample {
	public static void main(String[] args) {
		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

		// Generates input data.
		DataStream<Row> inputStream =
			env.fromElements(Row.of("Test for tokenization."), Row.of("Te,st. punct"));
		Table inputTable = tEnv.fromDataStream(inputStream).as("input");

		// Creates a RegexTokenizer object and initializes its parameters.
		RegexTokenizer regexTokenizer =
			new RegexTokenizer()
				.setInputCol("input")
				.setOutputCol("output")
				.setPattern("\\w+|\\p{Punct}");

		// Uses the Tokenizer object for feature transformations.
		Table outputTable = regexTokenizer.transform(inputTable)[0];

		// Extracts and displays the results.
		for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
			Row row = it.next();

			String inputValue = (String) row.getField(regexTokenizer.getInputCol());
			String[] outputValues = (String[]) row.getField(regexTokenizer.getOutputCol());

			System.out.printf(
				"Input Value: %s \tOutput Values: %s\n",
				inputValue, Arrays.toString(outputValues));
		}
	}
}


```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a RegexTokenizer instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.regextokenizer import RegexTokenizer
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()

t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_data_table = t_env.from_data_stream(
    env.from_collection([
        ('Test for tokenization.',),
        ('Te,st. punct',),
    ],
        type_info=Types.ROW_NAMED(
            ['input'],
            [Types.STRING()])))

# Creates a RegexTokenizer object and initializes its parameters.
regex_tokenizer = RegexTokenizer() \
    .set_input_col("input") \
    .set_output_col("output")

# Uses the Tokenizer object for feature transformations.
output = regex_tokenizer.transform(input_data_table)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(regex_tokenizer.get_input_col())]
    output_value = result[field_names.index(regex_tokenizer.get_output_col())]
    print('Input Values: ' + str(input_value) + '\tOutput Value: ' + str(output_value))

```

{{< /tab>}}

{{< /tabs>}}
