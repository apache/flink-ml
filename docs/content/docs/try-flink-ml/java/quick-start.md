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
Learning Model and uses it to provide prediction service.

## Help, I’m Stuck!

If you get stuck, check out the [community support
resources](https://flink.apache.org/gettinghelp.html). In particular, Apache
Flink's [user mailing
list](https://flink.apache.org/community.html#mailing-lists) is consistently
ranked as one of the most active of any Apache project and a great way to get
help quickly.

## Prerequisites

Make sure Java 8 or a higher version has been installed in your local machine.
To check the Java version installed, type in your terminal:

```shell
$ java -version
```

## Download Flink

[Download 1.16 or a higher version of
Flink](https://flink.apache.org/downloads.html), then extract the archive:

```shell
$ tar -xzf flink-*.tgz
```

## Set Up Flink Environment Variables

After having downloaded Flink, please register `$FLINK_HOME` as an environment
variable into your local environment.

```bash
cd ${path_to_flink}
export FLINK_HOME=`pwd`
```

## Add Flink ML library to Flink's library folder

You need to copy Flink ML's library files to Flink's folder for proper
initialization. 

{{< stable >}}

Please [download the correponding binary
release](https://flink.apache.org/downloads.html) of Flink ML, then extract the
archive:

```shell
tar -xzf flink-ml-*.tgz
```

Then you may copy the extracted library files to Flink's folder with the
following commands.

```shell
cd ${path_to_flink_ml}
cp ./lib/*.jar $FLINK_HOME/lib/
```

{{< /stable >}} {{< unstable >}}

Please walk through this [guideline]({{< ref
"docs/development/build-and-install#build-and-install-java-sdk" >}}) to build
Flink ML's Java SDK. After that, you may copy the generated library files to
Flink's folder with the following commands.

```shell
cd ${path_to_flink_ml}
cp ./flink-ml-dist/target/flink-ml-*-bin/flink-ml*/lib/*.jar $FLINK_HOME/lib/
```

{{< /unstable >}}

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

