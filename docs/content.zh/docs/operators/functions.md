---
title: "Functions"
type: docs
weight: 2
aliases:
- /operators/functions.html
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

## Functions

Flink ML provides users with some built-in table functions for data
transformations. This page gives a brief overview of them. 

### vectorToArray

This function converts a column of Flink ML sparse/dense vectors into a column
of double arrays.

{{< tabs vectorToArray_examples >}}

{{< tab "Java">}}
```java
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.List;

import static org.apache.flink.ml.Functions.vectorToArray;
import static org.apache.flink.table.api.Expressions.$;

/** Simple program that converts a column of dense/sparse vectors into a column of double arrays. */
public class VectorToArrayExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input vector data.
        List<Vector> vectors =
                Arrays.asList(
                        Vectors.dense(0.0, 0.0),
                        Vectors.sparse(2, new int[] {1}, new double[] {1.0}));
        Table inputTable =
                tEnv.fromDataStream(env.fromCollection(vectors, VectorTypeInfo.INSTANCE))
                        .as("vector");

        // Converts each vector to a double array.
        Table outputTable = inputTable.select($("vector"), vectorToArray($("vector")).as("array"));

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            Vector vector = row.getFieldAs("vector");
            Double[] doubleArray = row.getFieldAs("array");
            System.out.printf(
                    "Input vector: %s\tOutput double array: %s\n",
                    vector, Arrays.toString(doubleArray));
        }
    }
}
```
{{< /tab>}}

{{< tab "Python">}}
```python
# Simple program that converts a column of dense/sparse vectors into a column of double arrays.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from pyflink.ml.linalg import Vectors, VectorTypeInfo

from pyflink.ml.functions import vector_to_array
from pyflink.table.expressions import col

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input vector data
vectors = [
    (Vectors.dense(0.0, 0.0),),
    (Vectors.sparse(2, [1], [1.0]),),
]
input_table = t_env.from_data_stream(
    env.from_collection(
        vectors,
        type_info=Types.ROW_NAMED(
            ['vector'],
            [VectorTypeInfo()])
    ))

# convert each vector to a double array
output_table = input_table.select(vector_to_array(col('vector')).alias('array'))

# extract and display the results
output_values = [x for x in
                 t_env.to_data_stream(output_table).map(lambda r: r).execute_and_collect()]

output_values.sort(key=lambda x: x[0])

field_names = output_table.get_schema().get_field_names()
for i in range(len(output_values)):
    vector = vectors[i][0]
    double_array = output_values[i][field_names.index("array")]
    print("Input vector: %s \t output double array: %s" % (vector, double_array))
```
{{< /tab>}}

{{< /tabs>}}

### arrayToVector

This function converts a column of arrays of numeric type into a column of
DenseVector instances.

{{< tabs arrayToVector_examples >}}

{{< tab "Java">}}
```java
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.List;

import static org.apache.flink.ml.Functions.arrayToVector;
import static org.apache.flink.table.api.Expressions.$;

/** Simple program that converts a column of double arrays into a column of dense vectors. */
public class ArrayToVectorExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input double array data.
        List<double[]> doubleArrays =
                Arrays.asList(new double[] {0.0, 0.0}, new double[] {0.0, 1.0});
        Table inputTable = tEnv.fromDataStream(env.fromCollection(doubleArrays)).as("array");

        // Converts each double array to a dense vector.
        Table outputTable = inputTable.select($("array"), arrayToVector($("array")).as("vector"));

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            Double[] doubleArray = row.getFieldAs("array");
            Vector vector = row.getFieldAs("vector");
            System.out.printf(
                    "Input double array: %s\tOutput vector: %s\n",
                    Arrays.toString(doubleArray), vector);
        }
    }
}
```
{{< /tab>}}

{{< tab "Python">}}
```python
# Simple program that converts a column of double arrays into a column of dense vectors.

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.ml.functions import array_to_vector
from pyflink.table import StreamTableEnvironment
from pyflink.table.expressions import col

# create a new StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()

# create a StreamTableEnvironment
t_env = StreamTableEnvironment.create(env)

# generate input double array data
double_arrays = [
    ([0.0, 0.0],),
    ([0.0, 1.0],),
]
input_table = t_env.from_data_stream(
    env.from_collection(
        double_arrays,
        type_info=Types.ROW_NAMED(
            ['array'],
            [Types.PRIMITIVE_ARRAY(Types.DOUBLE())])
    ))

# convert each double array to a dense vector
output_table = input_table.select(array_to_vector(col('array')).alias('vector'))

# extract and display the results
field_names = output_table.get_schema().get_field_names()

output_values = [x[field_names.index('vector')] for x in
                 t_env.to_data_stream(output_table).execute_and_collect()]

output_values.sort(key=lambda x: x.get(1))

for i in range(len(output_values)):
    double_array = double_arrays[i][0]
    vector = output_values[i]
    print("Input double array: %s \t output vector: %s" % (double_array, vector))
```
{{< /tab>}}

{{< /tabs>}}
