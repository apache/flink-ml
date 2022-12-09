---
title: "RandomSplitter"
weight: 1
type: docs
aliases:
- /operators/feature/randomsplitter.html
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

## RandomSplitter

An AlgoOperator which splits a table into N tables according to the given weights.

### Parameters

| Key     | Default      | Type     | Required | Description                                                                                                                                                  |
|:--------|:-------------|:---------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| weights | `[1.0, 1.0]` | Double[] | no       | The weights of data splitting.                                                                                                                               |
| seed    | `null`       | Long     | no       | The random seed. This parameter guarantees reproduciable output only when the paralleism is unchanged and each worker reads the same data in the same order. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.feature.randomsplitter.RandomSplitter;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that creates a RandomSplitter instance and uses it for data splitting. */
public class RandomSplitterExample {
	public static void main(String[] args) {
		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

		// Generates input data.
		DataStream<Row> inputStream =
			env.fromElements(
				Row.of(1, 10, 0),
				Row.of(1, 10, 0),
				Row.of(1, 10, 0),
				Row.of(4, 10, 0),
				Row.of(5, 10, 0),
				Row.of(6, 10, 0),
				Row.of(7, 10, 0),
				Row.of(10, 10, 0),
				Row.of(13, 10, 3));
		Table inputTable = tEnv.fromDataStream(inputStream).as("input");

		// Creates a RandomSplitter object and initializes its parameters.
		RandomSplitter splitter = new RandomSplitter().setWeights(4.0, 6.0);

		// Uses the RandomSplitter to split inputData.
		Table[] outputTables = splitter.transform(inputTable);

		// Extracts and displays the results.
		System.out.println("Split Result 1 (40%)");
		for (CloseableIterator<Row> it = outputTables[0].execute().collect(); it.hasNext(); ) {
			System.out.printf("%s\n", it.next());
		}
		System.out.println("Split Result 2 (60%)");
		for (CloseableIterator<Row> it = outputTables[1].execute().collect(); it.hasNext(); ) {
			System.out.printf("%s\n", it.next());
		}
	}
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a RandomSplitter instance and uses it for data splitting.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.lib.feature.randomsplitter import RandomSplitter
from pyflink.table import StreamTableEnvironment

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates input table.
input_table = t_env.from_data_stream(
    env.from_collection([
        (1, 10, 0),
        (1, 10, 0),
        (1, 10, 0),
        (4, 10, 0),
        (5, 10, 0),
        (6, 10, 0),
        (7, 10, 0),
        (10, 10, 0),
        (13, 10, 0)
    ],
        type_info=Types.ROW_NAMED(
            ['f0', 'f1', "f2"],
            [Types.INT(), Types.INT(), Types.INT()])))

# Creates a RandomSplitter object and initializes its parameters.
splitter = RandomSplitter().set_weights(4.0, 6.0)

# Uses the RandomSplitter to split the dataset.
output = splitter.transform(input_table)

# Extracts and displays the results.
print("Split Result 1 (40%)")
for result in t_env.to_data_stream(output[0]).execute_and_collect():
    print(str(result))

print("Split Result 2 (60%)")
for result in t_env.to_data_stream(output[1]).execute_and_collect():
    print(str(result))

```

{{< /tab>}}

{{< /tabs>}}
