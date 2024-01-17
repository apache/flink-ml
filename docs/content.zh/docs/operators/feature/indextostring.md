---
title: "IndexToString"
weight: 1
type: docs
aliases:
- /operators/feature/indextostring.html
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

## IndexToString

`IndexToStringModel` transforms input index column(s) to string column(s) using
the model data computed by StringIndexer. It is a reverse operation of
StringIndexerModel.

### Input Columns

| Param name | Type    | Default | Description                          |
| :--------- | :------ | :------ | :----------------------------------- |
| inputCols  | Integer | `null`  | Indices to be transformed to string. |

### Output Columns

| Param name | Type   | Default | Description          |
| :--------- | :----- | :------ | :------------------- |
| outputCols | String | `null`  | Transformed strings. |

### Parameters

Below are the parameters required by `StringIndexerModel`.

| Key        | Default | Type   | Required | Description          |
| ---------- | ------- | ------ | -------- | -------------------- |
| inputCols  | `null`  | String | yes      | Input column names.  |
| outputCols | `null`  | String | yes      | Output column names. |

### Examples

{{< tabs index_to_string_examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.stringindexer.IndexToStringModel;
import org.apache.flink.ml.feature.stringindexer.StringIndexerModelData;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/**
 * Simple program that creates an IndexToStringModelExample instance and uses it for feature
 * engineering.
 */
public class IndexToStringModelExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Creates model data for IndexToStringModel.
        StringIndexerModelData modelData =
                new StringIndexerModelData(
                        new String[][] {{"a", "b", "c", "d"}, {"-1.0", "0.0", "1.0", "2.0"}});
        Table modelTable = tEnv.fromDataStream(env.fromElements(modelData)).as("stringArrays");

        // Generates input data.
        DataStream<Row> predictStream = env.fromElements(Row.of(0, 3), Row.of(1, 2));
        Table predictTable = tEnv.fromDataStream(predictStream).as("inputCol1", "inputCol2");

        // Creates an indexToStringModel object and initializes its parameters.
        IndexToStringModel indexToStringModel =
                new IndexToStringModel()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setModelData(modelTable);

        // Uses the indexToStringModel object for feature transformations.
        Table outputTable = indexToStringModel.transform(predictTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            int[] inputValues = new int[indexToStringModel.getInputCols().length];
            String[] outputValues = new String[indexToStringModel.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = (int) row.getField(indexToStringModel.getInputCols()[i]);
                outputValues[i] = (String) row.getField(indexToStringModel.getOutputCols()[i]);
            }

            System.out.printf(
                    "Input Values: %s \tOutput Values: %s\n",
                    Arrays.toString(inputValues), Arrays.toString(outputValues));
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates an IndexToStringModelExample instance and uses it
# for feature engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.stringindexer import IndexToStringModel
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
predict_table = t_env.from_data_stream(
    env.from_collection([
        (0, 3),
        (1, 2),
    ],
        type_info=Types.ROW_NAMED(
            ['input_col1', 'input_col2'],
            [Types.INT(), Types.INT()])
    ))

# create an index-to-string model and initialize its parameters and model data
model_data_table = t_env.from_data_stream(
    env.from_collection([
        ([['a', 'b', 'c', 'd'], [-1., 0., 1., 2.]],),
    ],
        type_info=Types.ROW_NAMED(
            ['stringArrays'],
            [Types.OBJECT_ARRAY(Types.OBJECT_ARRAY(Types.STRING()))])
    ))

model = IndexToStringModel() \
    .set_input_cols('input_col1', 'input_col2') \
    .set_output_cols('output_col1', 'output_col2') \
    .set_model_data(model_data_table)

# use the index-to-string model for feature engineering
output = model.transform(predict_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in model.get_input_cols()]
output_values = [None for _ in model.get_input_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(model.get_input_cols())):
        input_values[i] = result[field_names.index(model.get_input_cols()[i])]
        output_values[i] = result[field_names.index(model.get_output_cols()[i])]
    print('Input Values: ' + str(input_values) + '\tOutput Values: ' + str(output_values))

```

{{< /tab>}}

{{< /tabs>}}
