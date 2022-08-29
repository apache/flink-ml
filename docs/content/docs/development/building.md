---
title: "Building Flink ML from Source"
weight: 999
type: docs
aliases:
- /development/building.html

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

# Building Flink ML from Source

This page covers how to build Flink ML from sources.

## Build Flink ML Java SDK

In order to build Flink ML you need the source code. Either [download the source
of a release](https://flink.apache.org/downloads.html) or [clone the git
repository](https://github.com/apache/flink-ml.git).

In addition, you need **Maven 3** and a **JDK** (Java Development Kit). Flink ML
requires **at least Java 8** to build.

To clone from git, enter:

```bash
git clone https://github.com/apache/flink-ml.git
```

The simplest way of building Flink ML is by running:

```bash
mvn clean install -DskipTests
```

This instructs [Maven](http://maven.apache.org/) (`mvn`) to first remove all
existing builds (`clean`) and then create a new Flink binary (`install`).

After the build finishes, you can acquire the build result in the following path
from the root directory of Flink ML:

```
./flink-ml-dist/target/flink-ml-*-bin/flink-ml*/
```

## Build Flink ML Python SDK

### Prerequisites

1. Building Flink ML Java SDK 

   If you want to build Flink ML's Python SDK that can be used for pip
   installation, you must first build the Java SDK, as described in the section
   above.

2. Python version(3.6, 3.7, or 3.8) is required
   ```shell
   $ python --version
   # the version printed here must be 3.6, 3.7 or 3.8
   ```

3. Install the dependencies with the following command:
   ```shell
   $ python -m pip install -r flink-ml-python/dev/dev-requirements.txt
   ```

### Installation

Then go to the root directory of Flink ML source code and run this command to
build the sdist package of `apache-flink-ml`:

```shell
cd flink-ml-python; python setup.py sdist; cd ..;
```

The sdist package of `apache-flink-ml` will be found under
`./flink-ml-python/dist/`. It could be installed as follows:

```shell
python -m pip install flink-ml-python/dist/*.tar.gz
```

