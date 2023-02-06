---
title: "ChiSqTest"
type: docs
aliases:
- /operators/stats/chisqtest.html
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

## ChiSqTest

Chi-square Test computes the statistics of independence of variables in a contingency table,
e.g., p-value, and DOF(degree of freedom) for each input feature. The contingency table is
constructed from the observed categorical values.

### Input Columns

| Param name  | Type   | Default      | Description            |
|:------------|:-------|:-------------|:-----------------------|
| featuresCol | Vector | `"features"` | Feature vector.        |
| labelCol    | Number | `"label"`    | Label of the features. |

### Output Columns

If the output result is not flattened, the output columns are as follows.

| Column name        | Type      | Description                                                                                                                                            |
|--------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| "pValues"          | Vector    | Probability of obtaining a test statistic result at least as extreme as the one that was actually observed, assuming that the null hypothesis is true. |
| "degreesOfFreedom" | Int Array | Degree of freedom of the hypothesis test.                                                                                                              |
| "statistics"       | Vector    | Test statistic.                                                                                                                                        |

If the output result is flattened, the output columns are as follows.

| Column name       | Type   | Description                                                                                                                                            |
|-------------------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| "featureIndex"    | Int    | Index of the feature in the input vectors.                                                                                                             |
| "pValue"          | Double | Probability of obtaining a test statistic result at least as extreme as the one that was actually observed, assuming that the null hypothesis is true. |
| "degreeOfFreedom" | Int    | Degree of freedom of the hypothesis test.                                                                                                              |
| "statistic"       | Double | Test statistic.                                                                                                                                        |

### Parameters

| Key         | Default      | Type    | Required | Description                                                                              |
|-------------|--------------|---------|----------|------------------------------------------------------------------------------------------|
| labelCol    | `"label"`    | String  | no       | Label column name.                                                                       |
| featuresCol | `"features"` | String  | no       | Features column name.                                                                    |
| flatten     | `false`      | Boolean | no       | If false, the returned table contains only a single row, otherwise, one row per feature. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.stats.chisqtest.ChiSqTest;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that creates a ChiSqTest instance and uses it for statistics. */
public class ChiSqTestExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        Table inputTable =
                tEnv.fromDataStream(
                                env.fromElements(
                                        Row.of(0., Vectors.dense(5, 1.)),
                                        Row.of(2., Vectors.dense(6, 2.)),
                                        Row.of(1., Vectors.dense(7, 2.)),
                                        Row.of(1., Vectors.dense(5, 4.)),
                                        Row.of(0., Vectors.dense(5, 1.)),
                                        Row.of(2., Vectors.dense(6, 2.)),
                                        Row.of(1., Vectors.dense(7, 2.)),
                                        Row.of(1., Vectors.dense(5, 4.)),
                                        Row.of(2., Vectors.dense(5, 1.)),
                                        Row.of(0., Vectors.dense(5, 2.)),
                                        Row.of(0., Vectors.dense(5, 2.)),
                                        Row.of(1., Vectors.dense(9, 4.)),
                                        Row.of(1., Vectors.dense(9, 3.))))
                        .as("label", "features");

        // Creates a ChiSqTest object and initializes its parameters.
        ChiSqTest chiSqTest =
                new ChiSqTest().setFlatten(true).setFeaturesCol("features").setLabelCol("label");

        // Uses the ChiSqTest object for statistics.
        Table outputTable = chiSqTest.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            System.out.printf(
                    "Feature Index: %s\tP Value: %s\tDegree of Freedom: %s\tStatistics: %s\n",
                    row.getField("featureIndex"),
                    row.getField("pValue"),
                    row.getField("degreeOfFreedom"),
                    row.getField("statistic"));
        }
    }
}
```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that creates a ChiSqTest instance and uses it for statistics.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.stats.chisqtest import ChiSqTest
from pyflink.table import StreamTableEnvironment

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input data
input_table = t_env.from_data_stream(
    env.from_collection([
        (0., Vectors.dense(5, 1.)),
        (2., Vectors.dense(6, 2.)),
        (1., Vectors.dense(7, 2.)),
        (1., Vectors.dense(5, 4.)),
        (0., Vectors.dense(5, 1.)),
        (2., Vectors.dense(6, 2.)),
        (1., Vectors.dense(7, 2.)),
        (1., Vectors.dense(5, 4.)),
        (2., Vectors.dense(5, 1.)),
        (0., Vectors.dense(5, 2.)),
        (0., Vectors.dense(5, 2.)),
        (1., Vectors.dense(9, 4.)),
        (1., Vectors.dense(9, 3.))
    ],
        type_info=Types.ROW_NAMED(
            ['label', 'features'],
            [Types.DOUBLE(), DenseVectorTypeInfo()]))
)

# create a ChiSqTest object and initialize its parameters
chi_sq_test = ChiSqTest().set_flatten(True)

# use the ChiSqTest object for statistics
output = chi_sq_test.transform(input_table)[0]

# extract and display the results
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    print("Feature Index: %s\tP Value: %s\tDegree of Freedom: %s\tStatistics: %s" %
          (result[field_names.index('featureIndex')], result[field_names.index('pValue')],
           result[field_names.index('degreeOfFreedom')], result[field_names.index('statistic')]))

```

{{< /tab>}}

{{< /tabs>}}

