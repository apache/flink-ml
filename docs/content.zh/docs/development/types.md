---
title: "Data Types"
weight: 3
type: docs
aliases:
- /development/types.html
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

# Data Types

Flink ML supports all data types that have been supported by Flink Table API, as
well as data types listed in sections below.

## Vector

Flink ML provides support for vectors of double values. A `Vector` in Flink ML
can be either dense(`DenseVector`) or sparse(`SparseVector`), depending on how
users create them accordig to the vector's sparsity. Each vector is initialized
with a fixed size and users may get or set the double value of any 0-based index
location in the vector.

Flink ML also has a class named `Vectors` providing utility methods for
instantiating vectors.

{{< tabs vector >}}

{{< tab "Java">}}
```java
int n = 4;
int[] indices = new int[] {0, 2, 3};
double[] values = new double[] {0.1, 0.3, 0.4};

SparseVector vector = Vectors.sparse(n, indices, values);
```
{{< /tab>}}

{{< tab "Python">}}
```python
# Create a dense vector of 64-bit floats from a Python list or numbers.
>>> Vectors.dense([1, 2, 3])
DenseVector([1.0, 2.0, 3.0])
>>> Vectors.dense(1.0, 2.0)
DenseVector([1.0, 2.0])

# Create a sparse vector, using either a dict, a list of (index, value) pairs, or two separate
# arrays of indices and values.

>>> Vectors.sparse(4, {1: 1.0, 3: 5.5})
SparseVector(4, {1: 1.0, 3: 5.5})
>>> Vectors.sparse(4, [(1, 1.0), (3, 5.5)])
SparseVector(4, {1: 1.0, 3: 5.5})
>>> Vectors.sparse(4, [1, 3], [1.0, 5.5])
SparseVector(4, {1: 1.0, 3: 5.5})
```
{{< /tab>}}
{{< /tabs>}}