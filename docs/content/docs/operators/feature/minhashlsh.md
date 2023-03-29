---
title: "MinHashLSH"
weight: 1
type: docs
aliases:
- /operators/feature/minhashlsh.html
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

## MinHashLSH

MinHashLSH is a Locality Sensitive Hashing (LSH) scheme for Jaccard distance metric.
The input features are sets of natural numbers represented as non-zero indices of vectors,
either dense vectors or sparse vectors. Typically, sparse vectors are more efficient.

In addition to transforming input feature vectors to multiple hash values, the MinHashLSH 
model also supports approximate nearest neighbors search within a dataset regarding a key 
vector and approximate similarity join between two datasets.

### Input Columns

| Param name | Type   | Default   | Description            |
|:-----------|:-------|:----------|:-----------------------|
| inputCol   | Vector | `"input"` | Features to be mapped. |

### Output Columns

| Param name | Type          | Default    | Description  |
|:-----------|:--------------|:-----------|:-------------|
| outputCol  | DenseVector[] | `"output"` | Hash values. |

### Parameters

Below are the parameters required by `MinHashLSHModel`.

| Key                     | Default    | Type    | Required | Description                                                        |
|-------------------------|------------|---------|----------|--------------------------------------------------------------------|
| inputCol                | `"input"`  | String  | no       | Input column name.                                                 |
| outputCol               | `"output"` | String  | no       | Output column name.                                                |

`MinHashLSH` needs parameters above and also below.

| Key                     | Default    | Type    | Required | Description                                                        |
|-------------------------|------------|---------|----------|--------------------------------------------------------------------|
| seed                    | `null`     | Long    | no       | The random seed.                                                   |
| numHashTables           | `1`        | Integer | no       | Default number of hash tables, for OR-amplification.               |
| numHashFunctionPerTable | `1`        | Integer | no       | Default number of hash functions per table, for AND-amplification. |

### Examples

{{< tabs examples >}}

{{< tab "Java">}}

```java
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.ml.feature.lsh.MinHashLSH;
import org.apache.flink.ml.feature.lsh.MinHashLSHModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;

import java.util.Arrays;
import java.util.List;

import static org.apache.flink.table.api.Expressions.$;

/**
 * Simple program that trains a MinHashLSH model and uses it for approximate nearest neighbors and
 * similarity join.
 */
public class MinHashLSHExample {
    public static void main(String[] args) throws Exception {

        // Creates a new StreamExecutionEnvironment.
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Creates a StreamTableEnvironment.
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates two datasets.
        Table dataA =
                tEnv.fromDataStream(
                        env.fromCollection(
                                Arrays.asList(
                                        Row.of(
                                                0,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {0, 1, 2},
                                                        new double[] {1., 1., 1.})),
                                        Row.of(
                                                1,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {2, 3, 4},
                                                        new double[] {1., 1., 1.})),
                                        Row.of(
                                                2,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {0, 2, 4},
                                                        new double[] {1., 1., 1.}))),
                                Types.ROW_NAMED(
                                        new String[] {"id", "vec"},
                                        Types.INT,
                                        TypeInformation.of(SparseVector.class))));

        Table dataB =
                tEnv.fromDataStream(
                        env.fromCollection(
                                Arrays.asList(
                                        Row.of(
                                                3,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {1, 3, 5},
                                                        new double[] {1., 1., 1.})),
                                        Row.of(
                                                4,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {2, 3, 5},
                                                        new double[] {1., 1., 1.})),
                                        Row.of(
                                                5,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {1, 2, 4},
                                                        new double[] {1., 1., 1.}))),
                                Types.ROW_NAMED(
                                        new String[] {"id", "vec"},
                                        Types.INT,
                                        TypeInformation.of(SparseVector.class))));

        // Creates a MinHashLSH estimator object and initializes its parameters.
        MinHashLSH lsh =
                new MinHashLSH()
                        .setInputCol("vec")
                        .setOutputCol("hashes")
                        .setSeed(2022)
                        .setNumHashTables(5);

        // Trains the MinHashLSH model.
        MinHashLSHModel model = lsh.fit(dataA);

        // Uses the MinHashLSH model for transformation.
        Table output = model.transform(dataA)[0];

        // Extracts and displays the results.
        List<String> fieldNames = output.getResolvedSchema().getColumnNames();
        for (Row result :
                (List<Row>) IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect())) {
            Vector inputValue = result.getFieldAs(fieldNames.indexOf(lsh.getInputCol()));
            DenseVector[] outputValue = result.getFieldAs(fieldNames.indexOf(lsh.getOutputCol()));
            System.out.printf(
                    "Vector: %s \tHash values: %s\n", inputValue, Arrays.toString(outputValue));
        }

        // Finds approximate nearest neighbors of the key.
        Vector key = Vectors.sparse(6, new int[] {1, 3}, new double[] {1., 1.});
        output = model.approxNearestNeighbors(dataA, key, 2).select($("id"), $("distCol"));
        for (Row result :
                (List<Row>) IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect())) {
            int idValue = result.getFieldAs(fieldNames.indexOf("id"));
            double distValue = result.getFieldAs(result.getArity() - 1);
            System.out.printf("ID: %d \tDistance: %f\n", idValue, distValue);
        }

        // Approximately finds pairs from two datasets with distances smaller than the threshold.
        output = model.approxSimilarityJoin(dataA, dataB, .6, "id");
        for (Row result :
                (List<Row>) IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect())) {
            int idAValue = result.getFieldAs(0);
            int idBValue = result.getFieldAs(1);
            double distValue = result.getFieldAs(2);
            System.out.printf(
                    "ID from left: %d \tID from right: %d \t Distance: %f\n",
                    idAValue, idAValue, distValue);
        }
    }
}

```

{{< /tab>}}

{{< tab "Python">}}

```python
# Simple program that trains a MinHashLSH model and uses it for approximate nearest neighbors 
# and similarity join.


from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.linalg import Vectors, SparseVectorTypeInfo
from pyflink.ml.feature.lsh import MinHashLSH

# Creates a new StreamExecutionEnvironment.
env = StreamExecutionEnvironment.get_execution_environment()

# Creates a StreamTableEnvironment.
t_env = StreamTableEnvironment.create(env)

# Generates two datasets.
data_a = t_env.from_data_stream(
    env.from_collection([
        (0, Vectors.sparse(6, [0, 1, 2], [1., 1., 1.])),
        (1, Vectors.sparse(6, [2, 3, 4], [1., 1., 1.])),
        (2, Vectors.sparse(6, [0, 2, 4], [1., 1., 1.])),
    ], type_info=Types.ROW_NAMED(['id', 'vec'], [Types.INT(), SparseVectorTypeInfo()])))

data_b = t_env.from_data_stream(
    env.from_collection([
        (3, Vectors.sparse(6, [1, 3, 5], [1., 1., 1.])),
        (4, Vectors.sparse(6, [2, 3, 5], [1., 1., 1.])),
        (5, Vectors.sparse(6, [1, 2, 4], [1., 1., 1.])),
    ], type_info=Types.ROW_NAMED(['id', 'vec'], [Types.INT(), SparseVectorTypeInfo()])))

# Creates a MinHashLSH estimator object and initializes its parameters.
lsh = MinHashLSH() \
    .set_input_col('vec') \
    .set_output_col('hashes') \
    .set_seed(2022) \
    .set_num_hash_tables(5)

# Trains the MinHashLSH model.
model = lsh.fit(data_a)

# Uses the MinHashLSH model for transformation.
output = model.transform(data_a)[0]

# Extracts and displays the results.
field_names = output.get_schema().get_field_names()
for result in t_env.to_data_stream(output).execute_and_collect():
    input_value = result[field_names.index(lsh.get_input_col())]
    output_value = result[field_names.index(lsh.get_output_col())]
    print(f'Vector: {input_value} \tHash Values: {output_value}')

# Finds approximate nearest neighbors of the key.
key = Vectors.sparse(6, [1, 3], [1., 1.])
output = model.approx_nearest_neighbors(data_a, key, 2).select("id, distCol")
for result in t_env.to_data_stream(output).execute_and_collect():
    id_value = result[field_names.index("id")]
    dist_value = result[-1]
    print(f'ID: {id_value} \tDistance: {dist_value}')

# Approximately finds pairs from two datasets with distances smaller than the threshold.
output = model.approx_similarity_join(data_a, data_b, .6, "id")
for result in t_env.to_data_stream(output).execute_and_collect():
    id_a_value, id_b_value, dist_value = result
    print(f'ID from left: {id_a_value} \tID from right: {id_b_value} \t Distance: {dist_value}')

```

{{< /tab>}}

{{< /tabs>}}
