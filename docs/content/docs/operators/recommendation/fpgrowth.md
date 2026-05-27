---
title: "Swing"
type: docs
aliases:

- /operators/recommendation/swing.html

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

## FPGrowth

An AlgoOperator which implements the FPGrowth algorithm.

FPGrowth is an algorithm for frequent pattern mining. FP growth algorithm represents the database in the form of a
tree called a frequent pattern tree or FP tree.

Ignore NULL values and empty sequence in the feature column during <i>transform()</i>.

Use distinct elements from a sequence to mine frequent pattern.

See
<a href="http://dx.doi.org/10.1145/335191.335372">
Han et al., Mining frequent patterns without candidate generation</a>,
<a href="https://doi.org/10.1145/1454008.1454027">
Li et al., PFP Parallel FP-growth for query recommendation</a> and
<a href="https://dl.acm.org/doi/abs/10.1145/1133905.1133907">
Borgelt C. An Implementation of the FP-growth Algorithm</a> for more information.

### Input Columns

| Param name | Type   | Default   | Description                                |
|:-----------|:-------|:----------|:-------------------------------------------|
| itemsCol   | String | `"items"` | Items sequence. (e.g. "item1,item2,item3") |

### Structure of Output Table

#### Frequent Pattern Table

| Name          | Type            | Description                                              |
|:--------------|:----------------|:---------------------------------------------------------|
| items         | String          | Frequent pattern.                                        |
| support_count | Long            | Number of occurrences of the frequent pattern.             |
| item_count    | Long            | Number of elements in the frequent pattern.              |

#### Association Rule Table

| Name      | Type   | Description                                    |
|:----------|:-------|:-----------------------------------------------|
| rule   | String | Association rule. (e.g. "item1,item2=>item3")  |
| item_count | Double   | Number of elements in the association rule.    |
| lift   | Double   | Lift.                                          |
| support_percent   | Double   | Support (frequency of the association rule).   |
| confidence_percent   | Double | Confidence.                                    |
| transaction_count   | Long   | Number of occurrences of the association rule. |

### Parameters

Below are the parameters required by `FPGrowth`.

| Key                    | Default   | Type    | Required | Description                                                                           |
|:-----------------------|:----------|:--------|:---------|:--------------------------------------------------------------------------------------|
| itemsCol               | `"items"` | String  | no       | Item sequence column name.                                                            |
| fieldDelimiter         | `","`     | String  | no       | Field delimiter of item sequence.                                                     |
| minLift                | `1.0`     | Double  | no       | Minimal lift level for association rules.                                             |
| minConfidence                       | `0.6`     | Double | no       | Minimal confidence level for association rules.                                       |
| minSupport        | `0.02`    | Double | no       | Minimal support percent,                                                              |
| minSupportCount        | `-1`      | Double | no       | Minimal support count. MIN_ITEM_COUNT has no effect when less than or equal to 0      |
| maxPatternLength                 | `10`      | Integer | no       | Max frequent pattern length.                                                          |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
package org.apache.flink.ml.examples.recommendation;

import org.apache.flink.ml.recommendation.fpgrowth.FPGrowth;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/**
 * Simple program that creates a Swing instance and uses it to generate recommendations for items.
 */
public class FPGrowthExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(""),
                        Row.of("A,B,C,D"),
                        Row.of("B,C,E"),
                        Row.of("A,B,C,E"),
                        Row.of("B,D,E"),
                        Row.of("A,B,C,D,A"));

        Table inputTable = tEnv.fromDataStream(inputStream).as("items");

        // Creates a FPGrowth object and initializes its parameters.
        FPGrowth fpg = new FPGrowth().setMinSupportCount(3);

        // Transforms the data.
        Table[] outputTable = fpg.transform(inputTable);

        // Extracts and displays the frequent patterns.
        for (CloseableIterator<Row> it = outputTable[0].execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            String pattern = row.getFieldAs(0);
            Long support = row.getFieldAs(1);
            Long itemCount = row.getFieldAs(2);

            System.out.printf("pattern: %d, support count: %d, item_count:%d\n",pattern, support, itemCount);
        }

        // Extracts and displays the association rules.
        for (CloseableIterator<Row> it = outputTable[1].execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            String rule = row.getFieldAs(0);
            Double lift = row.getFieldAs(2);
            Double support = row.getFieldAs(3);
            Double confidence_percent = row.getFieldAs(4);

            System.out.printf("rule: %d, list: %f, support:%f, confidence:%f\n",rule, lift, support, confidence_percent);
        }
    }
}


```

{{< /tab>}}

{{< tab "Python">}}

```python

# Simple program that creates a FPGrowth instance and gives recommendations for items.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.recommendation.fpgrowth import FPGrowth

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_table = t_env.from_data_stream(
    env.from_collection([
        ("A,B,C,D",),
        ("B,C,E",),
        ("A,B,C,E",),
        ("B,D,E",),
        ("A,B,C,D",)
    ],
        type_info=Types.ROW_NAMED(
            ['items'],
            [Types.STRING()])
    ))

# Creates a fpgrowth object and initialize its parameters.
fpg = FPGrowth().set_min_support(0.6)

# Transforms the data to fpgrowth algorithm result.
output_table = fpg.transform(input_table)

# Extracts and display the results.
pattern_result_names = output_table[0].get_schema().get_field_names()
rule_result_names = output_table[1].get_schema().get_field_names()

patterns = t_env.to_data_stream(output_table[0]).execute_and_collect()
rules = t_env.to_data_stream(output_table[1]).execute_and_collect()

print("|\t"+"\t|\t".join(pattern_result_names)+"\t|")
for result in patterns:
    print(f'|\t{result[0]}\t|\t{result[1]}\t|\t{result[2]}\t|')
print("|\t"+" | ".join(rule_result_names)+"\t|")
for result in rules:
    print(f'|\t{result[0]}\t|\t{result[1]}\t|\t{result[2]}\t|\t{result[3]}'
          + f'\t|\t{result[4]}\t|\t{result[5]}\t|')

```

{{< /tab>}}

{{< /tabs>}}
