---
title: "Quick Start"
weight: 1
type: docs
aliases:
- /try-flink-ml/java/quick-start.html
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

### Install Flink

Please make sure Flink 1.15 or higher version has been installed in your local
environment. You can refer to the [local
installation](https://nightlies.apache.org/flink/flink-docs-master/docs/try-flink/local_installation/)
instruction on Flink's document website for how to achieve this.

### Set Up Flink Environment Variables

After having installed Flink, please register `$FLINK_HOME` as an environment
variable into your local environment.

```bash
cd ${path_to_flink}
export FLINK_HOME=`pwd`
```


[//]: # (TODO: Add instructions to download binary distribution when release is
    available)
### Build Flink ML library

In order to use Flink ML's CLI you need to have the latest binary distribution
of Flink ML. You can acquire the distribution by building Flink ML's source code
locally with the following command.

```bash
cd ${path_to_flink_ml}
mvn clean package -DskipTests
cd ./flink-ml-dist/target/flink-ml-*-bin/flink-ml*/
```

### Add Flink ML binaries to Flink

You need to copy Flink ML's binary distribution files to Flink's folder for
proper initialization. Please run the following command from Flink ML's binary
distribution's folder.

```bash
cp ./lib/*.jar $FLINK_HOME/lib/
```

## Run Flink ML example job

Please start a Flink standalone cluster in your local environment with the
following command.

```bash
$FLINK_HOME/bin/start-cluster.sh
```

You should be able to navigate to the web UI at
[localhost:8081](http://localhost:8081/) to view the Flink dashboard and see
that the cluster is up and running.

Then you may submit Flink ML examples to the cluster as follows.

```bash
$FLINK_HOME/bin/flink run -c org.apache.flink.ml.examples.clustering.KMeansExample $FLINK_HOME/lib/flink-ml-examples*.jar
```

The command above would submit and execute Flink ML's `KMeansExample` job. There
are also example jobs for other Flink ML algorithms, and you can find them in
`flink-ml-examples` module.

A sample output in your terminal is as follows.

```
Features: [9.0, 0.0]    Cluster ID: 1
Features: [0.3, 0.0]    Cluster ID: 0
Features: [0.0, 0.3]    Cluster ID: 0
Features: [9.6, 0.0]    Cluster ID: 1
Features: [0.0, 0.0]    Cluster ID: 0
Features: [9.0, 0.6]    Cluster ID: 1

```

Now you have successfully run a Flink ML job.

