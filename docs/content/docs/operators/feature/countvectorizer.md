---
title: "Count Vectorizer"
weight: 1
type: docs
aliases:
- /operators/feature/countvectorizer.html
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
specific language governing permissions dand limitations
under the License.
-->

## Count Vectorizer

CountVectorizer is an algorithm that converts a collection of text
documents to vectors of token counts. When an a-priori dictionary is not 
available, CountVectorizer can be used as an estimator to extract the 
vocabulary, and generates a CountVectorizerModel. The model produces sparse
representations for the documents over the vocabulary, which can then be 
passed to other algorithms like LDA.

### Input Columns

| Param name | Type     | Default   | Description         |
|:-----------|:---------|:----------|:--------------------|
| inputCol   | String[] | `"input"` | Input string array. |

### Output Columns

| Param name | Type         | Default    | Description             |
|:-----------|:-------------|:-----------|:------------------------|
| outputCol  | SparseVector | `"output"` | Vector of token counts. |

### Parameters

Below are the parameters required by `CountVectorizerModel`.

| Key        | Default    | Type    | Required | Description                                                                                                                                                                                                                                                                                                                                     |
|------------|------------|---------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| inputCol   | `"input"`  | String  | no       | Input column name.                                                                                                                                                                                                                                                                                                                              |
| outputCol  | `"output"` | String  | no       | Output column name.                                                                                                                                                                                                                                                                                                                             |
| minTF      | `1.0`      | Double  | no       | Filter to ignore rare words in a document. For each document, terms with frequency/count less than the given threshold are ignored. If this is an integer >= 1, then this specifies a count (of times the term must appear in the document); if this is a double in [0,1), then this specifies a fraction (out of the document's token count).  |
| binary     | `false`    | Boolean | no       | Binary toggle to control the output vector values. If True, all nonzero counts (after minTF filter applied) are set to 1.0.                                                                                                                                                                                                                     |

`CountVectorizer` needs parameters above and also below.

| Key            | Default    | Type     | Required | Description                                                                                                                                                                                                                                                                                                                                                                                  |
|:---------------|:-----------|:---------|:---------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| vocabularySize | `2^18`     | Integer  | no       | Max size of the vocabulary. CountVectorizer will build a vocabulary that only considers the top vocabulary size terms ordered by term frequency across the corpus.                                                                                                                                                                                                                           |
| minDF          | `1.0`      | Double   | no       | Specifies the minimum number of different documents a term must appear in to be included in the vocabulary. If this is an integer >= 1, this specifies the number of documents the term must appear in; if this is a double in [0,1), then this specifies the fraction of documents.                                                                                                         |
| maxDF          | `2^63 - 1` | Double   | no       | Specifies the maximum number of different documents a term could appear in to be included in the vocabulary. A term that appears more than the threshold will be ignored. If this is an integer >= 1, this specifies the maximum number of documents the term could appear in; if this is a double in [0,1), then this specifies the maximum fraction of documents the term could appear in. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.countvectorizer.CountVectorizer;
import org.apache.flink.ml.feature.countvectorizer.CountVectorizerModel;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/**
 * Simple program that trains a {@link CountVectorizer} model and uses it for feature engineering.
 */
public class CountVectorizerExample {

    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> dataStream =
                env.fromElements(
                        Row.of((Object) new String[] {"a", "c", "b", "c"}),
                        Row.of((Object) new String[] {"c", "d", "e"}),
                        Row.of((Object) new String[] {"a", "b", "c"}),
                        Row.of((Object) new String[] {"e", "f"}),
                        Row.of((Object) new String[] {"a", "c", "a"}));
        Table inputTable = tEnv.fromDataStream(dataStream).as("input");

        // Creates an CountVectorizer object and initialize its parameters
        CountVectorizer countVectorizer = new CountVectorizer();

        // Trains the CountVectorizer model
        CountVectorizerModel model = countVectorizer.fit(inputTable);

        // Uses the CountVectorizer model for predictions.
        Table outputTable = model.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            String[] inputValue = (String[]) row.getField(countVectorizer.getInputCol());
            SparseVector outputValue = (SparseVector) row.getField(countVectorizer.getOutputCol());
            System.out.printf(
                    "Input Value: %-15s \tOutput Value: %s\n",
                    Arrays.toString(inputValue), outputValue.toString());
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python

# Simple program that creates an CountVectorizer instance and uses it for feature
# engineering.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.lib.feature.countvectorizer import CountVectorizer
from pyflink.table import StreamTableEnvironment

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input training and prediction data.
input_table = t_env.from_data_stream(
    env.from_collection([
        (1, ['a', 'c', 'b', 'c'],),
        (2, ['c', 'd', 'e'],),
        (3, ['a', 'b', 'c'],),
        (4, ['e', 'f'],),
        (5, ['a', 'c', 'a'],),
    ],
        type_info=Types.ROW_NAMED(
            ['id', 'input', ],
            [Types.INT(), Types.OBJECT_ARRAY(Types.STRING())])
    ))

# Creates an CountVectorizer object and initializes its parameters.
count_vectorizer = CountVectorizer()

# Trains the CountVectorizer Model.
model = count_vectorizer.fit(input_table)

# Uses the CountVectorizer Model for predictions.
output = model.transform(input_table)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_index = field_names.index(count_vectorizer.get_input_col())
    output_index = field_names.index(count_vectorizer.get_output_col())
    print('Input Value: %-20s Output Value: %10s' %
          (str(result[input_index]), str(result[output_index])))

```

{{< /tab>}}

{{< /tabs>}}
