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

## Swing

An AlgoOperator which implements the Swing algorithm.

Swing is an item recall algorithm. The topology of user-item graph usually can be described as
user-item-user or item-user-item, which are like 'swing'. For example, if both user <em>u</em> and user <em>v</em>
have purchased the same commodity <em>i</em>, they will form a relationship diagram similar to a swing. If
<em>u</em> and <em>v</em> have purchased commodity <em>j</em> in addition to <em>i</em>, it is supposed <em>i</em>
and <em>j</em> are similar. 

See "<a href="https://arxiv.org/pdf/2010.05525.pdf">Large Scale Product Graph Construction for Recommendation in
E-commerce</a>" by Xiaoyong Yang, Yadong Zhu and Yi Zhang.

### Input Columns

| Param name | Type | Default  | Description |
|:-----------|:-----|:---------|:------------|
| itemCol    | Long | `"item"` | Item id.    |
| userCol    | Long | `"user"` | User id.    |
### Output Columns

| Param name | Type   | Default    | Description                                                                                    |
|:-----------|:-------|:-----------|:-----------------------------------------------------------------------------------------------|
| itemCol    | Long   | `"item"`   | Item id.                                                                                       |
| outputCol  | String | `"output"` | Top k similar items and their corresponding scores. (e.g. "item_1,0.9;item_2,0.7;item_3,0.35") |

### Parameters

Below are the parameters required by `Swing`.

| Key               | Default    | Type    | Required | Description                                                                                                                                                                                                                                                                                                                                                                                               |
|:------------------|:-----------|:--------|:---------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| userCol           | `"user"`   | String  | no       | User column name.                                                                                                                                                                                                                                                                                                                                                                                         |
| itemCol           | `"item"`   | String  | no       | Item column name.                                                                                                                                                                                                                                                                                                                                                                                         |
| maxUserNumPerItem | `1000`     | Integer | no       | The max number of user(purchasers) for each item. If the number of user is larger than this value, then only maxUserNumPerItem users will be sampled and considered in the computation of similarity between two items.                                                                                                                                                                                   |
| k                 | `100`      | Integer | no       | The max number of similar items to output for each item.                                                                                                                                                                                                                                                                                                                                                  |
| minUserBehavior   | `10`       | Integer | no       | The min number of items for a user purchases. If the items purchased by a user is smaller than this value, then this user is filtered out while gathering data. This can affect the speed of the computation. Set minUserBehavior larger in case the swing recommendation progresses very slowly.                                                                                                         |
| maxUserBehavior   | `1000`     | Integer | no       | The max number of items for a user purchases. If the items purchased by a user is larger than this value, then this user is filtered out while gathering data. This can affect the speed of the computation. Set maxUserBehavior smaller in case the swing recommendation progresses very slowly. The IllegalArgumentException is raised if the value of maxUserBehavior is smaller than minUserBehavior. |
| alpha1            | `15`       | Integer | no       | Smooth factor for number of users that have purchased one item. The higher alpha1 is, the less purchasing behavior contributes to the similarity score.                                                                                                                                                                                                                                                   |
| alpha2            | `0`        | Integer | no       | Smooth factor for number of users that have purchased the two target items. The higher alpha2 is, the less purchasing behavior contributes to the similarity score.                                                                                                                                                                                                                                       |
| beta              | `0.3`      | Double  | no       | Decay factor for number of users that have purchased one item. The higher beta is, the less purchasing behavior contributes to the similarity score.                                                                                                                                                                                                                                                      |
| outputCol         | `"output"` | String  | no       | Output column name.                                                                                                                                                                                                                                                                                                                                                                                       |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
package org.apache.flink.ml.examples.recommendation;

import org.apache.flink.ml.recommendation.swing.Swing;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/**
 * Simple program that creates a Swing instance and uses it to generate recommendations for items.
 */
public class SwingExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(0L, 10L),
                        Row.of(0L, 11L),
                        Row.of(0L, 12L),
                        Row.of(1L, 13L),
                        Row.of(1L, 12L),
                        Row.of(2L, 10L),
                        Row.of(2L, 11L),
                        Row.of(2L, 12L),
                        Row.of(3L, 13L),
                        Row.of(3L, 12L));

        Table inputTable = tEnv.fromDataStream(inputStream).as("user", "item");

        // Creates a Swing object and initializes its parameters.
        Swing swing = new Swing().setUserCol("user").setItemCol("item").setMinUserBehavior(1);

        // Transforms the data.
        Table[] outputTable = swing.transform(inputTable);

        // Extracts and displays the result of swing algorithm.
        for (CloseableIterator<Row> it = outputTable[0].execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            long mainItem = row.getFieldAs(0);
            String itemRankScore = row.getFieldAs(1);

            System.out.printf("item: %d, top-k similar items: %s\n", mainItem, itemRankScore);
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python

# Simple program that creates a Swing instance and gives recommendations for items.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.recommendation.swing import Swing

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input data.
input_table = t_env.from_data_stream(
    env.from_collection([
        (0, 10),
        (0, 11),
        (0, 12),
        (1, 13),
        (1, 12),
        (2, 10),
        (2, 11),
        (2, 12),
        (3, 13),
        (3, 12)
    ],
        type_info=Types.ROW_NAMED(
            ['user', 'item'],
            [Types.LONG(), Types.LONG()])
    ))

# Creates a swing object and initialize its parameters.
swing = Swing()
    .set_item_col('item')
    .set_user_col("user")
    .set_min_user_behavior(1)

# Transforms the data to Swing algorithm result.
output_table = swing.transform(input_table)

# Extracts and display the results.
field_names = output_table[0].get_schema().get_field_names()

results = t_env.to_data_stream(
    output_table[0]).execute_and_collect()

for result in results:
    main_item = result[field_names.index(swing.get_item_col())]
    item_rank_score = result[1]
    print(f'item: {main_item}, top-k similar items: {item_rank_score}')

```

{{< /tab>}}

{{< /tabs>}}
