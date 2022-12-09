---
title: "StopWordsRemover"
weight: 1
type: docs
aliases:
- /operators/feature/stopwordsremover.html
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

## StopWordsRemover

A feature transformer that filters out stop words from input.

Note: null values from input array are preserved unless adding null to stopWords
explicitly.

See Also: <a href="http://en.wikipedia.org/wiki/Stop_words">Stop words
(Wikipedia)</a>

### Input Columns

| Param name | Type     | Default | Description                                        |
|:-----------|:---------|:--------|:---------------------------------------------------|
| inputCols  | String[] | `null`  | Arrays of strings containing stop words to remove. |

### Output Columns

| Param name | Type     | Default | Description                                |
|:-----------|:---------|:--------|:-------------------------------------------|
| outputCols | String[] | `null`  | Arrays of strings with stop words removed. |

### Parameters

| Key           | Default                                            | Type     | Required | Description                                                                            |
|---------------|----------------------------------------------------|----------|----------|----------------------------------------------------------------------------------------|
| inputCols     | `null`                                             | String[] | yes      | Input column names.                                                                    |
| outputCols    | `null`                                             | String[] | yes      | Output column name.                                                                    |
| stopWords     | `StopWordsRemover.loadDefaultStopWords("english")` | String[] | no       | The words to be filtered out.                                                          |
| caseSensitive | `false`                                            | Boolean  | no       | Whether to do a case-sensitive comparison over the stop words.                         |
| locale        | `StopWordsRemover.getDefaultOrUS().toString()`     | String   | no       | Locale of the input for case insensitive matching. Ignored when caseSensitive is true. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.stopwordsremover.StopWordsRemover;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates a StopWordsRemover instance and uses it for feature engineering. */
public class StopWordsRemoverExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of((Object) new String[] {"test", "test"}),
                        Row.of((Object) new String[] {"a", "b", "c", "d"}),
                        Row.of((Object) new String[] {"a", "the", "an"}),
                        Row.of((Object) new String[] {"A", "The", "AN"}),
                        Row.of((Object) new String[] {null}),
                        Row.of((Object) new String[] {}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("input");

        // Creates a StopWordsRemover object and initializes its parameters.
        StopWordsRemover remover =
                new StopWordsRemover().setInputCols("input").setOutputCols("output");

        // Uses the StopWordsRemover object for feature transformations.
        Table outputTable = remover.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            String[] inputValues = row.getFieldAs("input");
            String[] outputValues = row.getFieldAs("output");

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
# Simple program that creates a StopWordsRemover instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.lib.feature.stopwordsremover import StopWordsRemover
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_table = t_env.from_data_stream(
    env.from_collection([
        (["test", "test"],),
        (["a", "b", "c", "d"],),
        (["a", "the", "an"],),
        (["A", "The", "AN"],),
        ([None],),
        ([],),
    ],
        type_info=Types.ROW_NAMED(
            ['input'],
            [Types.OBJECT_ARRAY(Types.STRING())])))

# create a StopWordsRemover object and initialize its parameters
remover = StopWordsRemover().set_input_cols('input').set_output_cols('output')

# use the StopWordsRemover for feature engineering
output_table = remover.transform(input_table)[0]

# extract and display the results
field_names = output_table.get_schema().get_field_names()
for result in t_env.to_data_stream(output_table).execute_and_collect():
    input_value = result[field_names.index('input')]
    output_value = result[field_names.index('output')]
    print('Input Value: ' + str(input_value) + '\tOutput Value: ' + str(output_value))
```

{{< /tab>}}

{{< /tabs>}}
