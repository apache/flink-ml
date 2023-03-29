---
title: "StringIndexer"
weight: 1
type: docs
aliases:
- /operators/feature/stringindexer.html
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

## StringIndexer

StringIndexer maps one or more columns (string/numerical value) of the input to
one or more indexed output columns (integer value). The output indices of two
data points are the same iff their corresponding input columns are the same. The
indices are in [0, numDistinctValuesInThisColumn].

IndexToStringModel transforms input index column(s) to string column(s) using
the model data computed by StringIndexer. It is a reverse operation of
StringIndexerModel.
### Input Columns

| Param name | Type          | Default | Description                            |
| :--------- | :------------ | :------ |:---------------------------------------|
| inputCols  | Number/String | `null`  | String/Numerical values to be indexed. |

### Output Columns

| Param name | Type   | Default | Description                         |
|:-----------|:-------|:--------|:------------------------------------|
| outputCols | Double | `null`  | Indices of string/numerical values. |

### Parameters

Below are the parameters required by `StringIndexerModel`.

| Key           | Default   | Type     | Required | Description                                                                    |
|---------------|-----------|----------|----------|--------------------------------------------------------------------------------|
| inputCols     | `null`    | String[] | yes      | Input column names.                                                            |
| outputCols    | `null`    | String[] | yes      | Output column names.                                                           |
| handleInvalid | `"error"` | String   | no       | Strategy to handle invalid entries. Supported values: 'error', 'skip', 'keep'. |

`StringIndexer` needs parameters above and also below.

| Key             | Default       | Type   | Required | Description                                                                                                                         |
|-----------------|---------------|--------|----------|-------------------------------------------------------------------------------------------------------------------------------------|
| stringOrderType | `"arbitrary"` | String | no       | How to order strings of each column. Supported values: 'arbitrary', 'frequencyDesc', 'frequencyAsc', 'alphabetDesc', 'alphabetAsc'. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.stringindexer.StringIndexer;
import org.apache.flink.ml.feature.stringindexer.StringIndexerModel;
import org.apache.flink.ml.feature.stringindexer.StringIndexerParams;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that trains a StringIndexer model and uses it for feature engineering. */
public class StringIndexerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of("a", 1.0),
                        Row.of("b", 1.0),
                        Row.of("b", 2.0),
                        Row.of("c", 0.0),
                        Row.of("d", 2.0),
                        Row.of("a", 2.0),
                        Row.of("b", 2.0),
                        Row.of("b", -1.0),
                        Row.of("a", -1.0),
                        Row.of("c", -1.0));
        Table trainTable = tEnv.fromDataStream(trainStream).as("inputCol1", "inputCol2");

        DataStream<Row> predictStream =
                env.fromElements(Row.of("a", 2.0), Row.of("b", 1.0), Row.of("c", 2.0));
        Table predictTable = tEnv.fromDataStream(predictStream).as("inputCol1", "inputCol2");

        // Creates a StringIndexer object and initializes its parameters.
        StringIndexer stringIndexer =
                new StringIndexer()
                        .setStringOrderType(StringIndexerParams.ALPHABET_ASC_ORDER)
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2");

        // Trains the StringIndexer Model.
        StringIndexerModel model = stringIndexer.fit(trainTable);

        // Uses the StringIndexer Model for predictions.
        Table outputTable = model.transform(predictTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            Object[] inputValues = new Object[stringIndexer.getInputCols().length];
            double[] outputValues = new double[stringIndexer.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = row.getField(stringIndexer.getInputCols()[i]);
                outputValues[i] = (double) row.getField(stringIndexer.getOutputCols()[i]);
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
# Simple program that trains a StringIndexer model and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.feature.stringindexer import StringIndexer
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input training and prediction data
train_table = t_env.from_data_stream(
    env.from_collection([
        ('a', 1.),
        ('b', 1.),
        ('b', 2.),
        ('c', 0.),
        ('d', 2.),
        ('a', 2.),
        ('b', 2.),
        ('b', -1.),
        ('a', -1.),
        ('c', -1.),
    ],
        type_info=Types.ROW_NAMED(
            ['input_col1', 'input_col2'],
            [Types.STRING(), Types.DOUBLE()])
    ))

predict_table = t_env.from_data_stream(
    env.from_collection([
        ('a', 2.),
        ('b', 1.),
        ('c', 2.),
    ],
        type_info=Types.ROW_NAMED(
            ['input_col1', 'input_col2'],
            [Types.STRING(), Types.DOUBLE()])
    ))

# create a string-indexer object and initialize its parameters
string_indexer = StringIndexer() \
    .set_string_order_type('alphabetAsc') \
    .set_input_cols('input_col1', 'input_col2') \
    .set_output_cols('output_col1', 'output_col2')

# train the string-indexer model
model = string_indexer.fit(train_table)

# use the string-indexer model for feature engineering
output = model.transform(predict_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
input_values = [None for _ in string_indexer.get_input_cols()]
output_values = [None for _ in string_indexer.get_input_cols()]
for result in t_env.to_data_stream(output).execute_and_collect():
    for i in range(len(string_indexer.get_input_cols())):
        input_values[i] = result[field_names.index(string_indexer.get_input_cols()[i])]
        output_values[i] = result[field_names.index(string_indexer.get_output_cols()[i])]
    print('Input Values: ' + str(input_values) + '\tOutput Values: ' + str(output_values))

```

{{< /tab>}}

{{< /tabs>}}
