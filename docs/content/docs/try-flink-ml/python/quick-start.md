---
title: "Quick Start"
weight: 1
type: docs
aliases:
- /try-flink-ml/python/quick-start.html

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

# Quick Start

This document provides a quick introduction to using Flink ML. Readers of this
document will be guided to submit a simple Flink job that trains a Machine
Learning Model and use it to provide prediction service.

## Prerequisites

Python version (3.6, 3.7, or 3.8) is required for Flink ML. Please run the
following command to make sure that it meets the requirements:

```shell
$ python --version
# the version printed here must be 3.6, 3.7 or 3.8
```

## Installation of Flink ML Python SDK

Flink ML Python SDK is available in
[PyPi](https://pypi.org/project/apache-flink-ml/) and can be installed as
follows:

{{< stable >}}

```bash
$ python -m pip install apache-flink-ml=={{< version >}}
```

{{< /stable >}} {{< unstable >}}

```bash
$ python -m pip install apache-flink-ml
```

{{< /unstable >}}

You can also build Flink ML Python SDK from sources by following the
[development guide]({{< ref "docs/development/building" >}}).

## Run Flink ML example job

After setting up Flink ML Python SDK, you can run a Flink ML example job as
follows.

```shell
$ python -m pyflink.examples.ml.clustering.kmeans_example
```

The command above would create a Flink mini-cluster and execute Flink ML’s
`kmeans_example` job. There are also example jobs for other Flink ML algorithms,
and you can find them in `pyflink.ml.examples` module.

A sample output in your terminal is as follows.

```
Features: [9.6,0.0]     Cluster Id: 0
Features: [9.0,0.6]     Cluster Id: 0
Features: [0.0,0.3]     Cluster Id: 1
Features: [0.0,0.0]     Cluster Id: 1
Features: [0.3,3.0]     Cluster Id: 1
Features: [9.0,0.0]     Cluster Id: 0

```

Now you have successfully run a Flink ML job.

