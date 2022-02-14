---
layout: post 
title:  "Apache Flink ML 2.0.0 Release Announcement"
date: 2022-01-07T08:00:00.000Z
type: docs
categories: news
authors:
- lindong:
  name: "Dong Lin"
- gaoyun:
  name: "Yun Gao"

excerpt: The Apache Flink community is excited to announce the release of Flink ML 2.0.0! This release involves a major refactor of the earlier Flink ML library and introduces major features that extend the Flink ML API and the iteration runtime, such as supporting stages with multi-input multi-output, graph-based stage composition, and a new stream-batch unified iteration library.

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

The Apache Flink community is excited to announce the release of Flink ML
2.0.0! Flink ML is a library that provides APIs and infrastructure for building
stream-batch unified machine learning algorithms, that can be easy-to-use and
performant with (near-) real-time latency.

This release involves a major refactor of the earlier Flink ML library and
introduces major features that extend the Flink ML API and the iteration
runtime, such as supporting stages with multi-input multi-output, graph-based
stage composition, and a new stream-batch unified iteration library. Moreover,
we added five algorithm implementations in this release, which is the start of
a long-term initiative to provide a large number of off-the-shelf algorithms in
Flink ML with state-of-the-art performance.

We believe this release is an important step towards extending Apache Flink to
a wide range of machine learning use cases, especially the real-time machine
learning scenarios.

We encourage you to [download the release](https://flink.apache.org/downloads.html) and share your feedback with
the community through the Flink [mailing lists](https://flink.apache.org/community.html#mailing-lists) or
[JIRA](https://issues.apache.org/jira/browse/flink)! We hope you like the new
release and we’d be eager to learn about your experience with it.

# Notable Features

## API and Infrastructure

### Supporting stages requiring multi-input multi-output

Stages in a machine learning workflow might take multiple inputs and return
multiple outputs. For example, a graph embedding algorithm might need to read
two tables, which represent the edge and node of the graph respectively. And a
workflow might need a stage that splits the input dataset into two output
datasets, for training and testing respectively.

With this capability, algorithm developers can assemble a machine learning
workflow as a directed acyclic graph (DAG) of pre-defined stages. And this
workflow can be configured and deployed without users knowing the
implementation details of this graph. This improvement could considerably
expand the applicability and usability of Flink ML.

### Supporting online learning with APIs exposing model data

In a native online learning scenario, we have a long-running job that keeps
processing training data and updating a machine learning model. And we could
have multiple jobs deployed in web servers which do online inference. It is
necessary to transmit the latest model data from the training job to those
inference jobs in (near-) real-time latency.

The traditional Estimator/Transformer paradigm does not provide APIs to expose
this model data in a streaming manner. Users have to repeatedly call fit() to
update model data. Although users might be able to update model data once every
few minutes, it is likely very inefficient, if not impossible, to update model
data once every few seconds with this approach.

With
[FLIP-173](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=184615783),
model data can be exposed as an unbounded stream via the getModelData() API.
Then algorithm users can transfer the model data to web servers in real-time
and use the up-to-date model data to do online inference.  This feature could
significantly strengthen Flink ML’s capability to support online learning
applications.

### Improved usability for managing parameters

We care a lot about usability and developer velocity in Flink ML. In this
release, we refactored and significantly simplified the experience of defining,
getting and setting parameters for algorithms. 

With
[FLIP-174](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=181311361),
parameters can be defined as static variables of an interface, and any
algorithm that implements the interface could inherit these variable
definitions without additional work. Commonly used parameter validators are
provided as part of the infrastructure.

### Tools for composing DAG of stages into a new stage

One of the most useful tool in the existing ML libraries (e.g. Scikit-learn,
Flink, Spark) is
[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html),
which allows users to compose an estimator from an ordered list of estimators
and transformers, without having to explicitly implement the fit/transform for
the estimator/transformer.

[FLIP-175](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=181311363)
extended this capability from pipeline to DAG. Users can now compose an
estimator from a DAG of estimator and transformers. This capability of
composition allows developers to slice a complex workflow into simpler modules
and re-use the modules across multiple workflows. We believe this capability
could significantly improve the experience of building and deploying complex
workflows using Flink ML.

## Stream-batch Unified Iteration Library

To support training machine learning algorithms and adjust the model parameters
dynamically based on the prediction result, it is necessary to have native
support for processing data iteratively. It is known that Flink uses DAG to
describe the process logic, thus we need to provide the iteration library on
top of Flink separately. Besides, since we need to support both offline
training and online training / adjustment, the iteration library should support
both streaming and batch cases. 

[FLIP-176](https://cwiki.apache.org/confluence/x/hAEBCw) implements a
stream-batch unified iteration library. It provides the function of
transmitting records back to the precedent operators and the ability to track
the progress of rounds inside the iteration. Users could directly use
DataStream API and Table API to express the execution logic inside the
iteration. Besides, the new iteration library also extends Flink’s
checkpointing mechanism to also support exactly-once failover for jobs using
iterations. 

## Python SDK

Nowadays many machine learning practitioners are used to developing machine
learning workflows in Python due to its ease-of-use and excellent ecosystem. To
meet the needs of these users, a Python package dedicated for Flink ML is
created starting from this release. The Python package currently provides APIs
similar to their Java counterparts for developing machine learning algorithms.

Users can install Flink ML Python package through pip using the following
command:

```bash
pip install apache-flink-ml
```

In the future we will enhance the Python SDK to enable its interoperability
with Flink ML’s Java library, for example, allowing users to express machine
learning workflows in Python, where workflows consist of a mixture of stages
from the Flink ML Java library as well as stages implemented in Python (e.g. a
TensorFlow program).

## Algorithm Library

Now that the Flink ML API re-design is done, we started the initiative to add
off-the-shelf algorithms in Flink ML. The release of Flink-ML 2.0.0 is closely
related to project Alink - an Apache Flink ecosystem project open sourced by
Alibaba. The connection between the Flink community and developers of the Alink
project dates back to 2017. The project Alink developers have a significant
contribution in designing the new Flink ML APIs, refactoring, optimizing and
migrating algorithms from Alink to Flink. Our long-term goal is to provide a
library of performant algorithms that are easy to use, debug and customize for
your needs.

We have implemented five algorithms in this release, i.e. logistic regression,
k-means, k-nearest neighbors, naive bayes and one-hot encoder. For now these
algorithms focus on validating the APIs and iteration runtime. In addition to
adding more and more algorithms, we will also stress test and optimize their
performance to make sure these algorithms have state-of-the-art performance.
Stay tuned!

# Related Work

## Flink ML project moved to a separate repository

To accelerate the development of Flink ML, the effort has moved to the new
repository [flink-ml](https://github.com/apache/flink-ml) under the Flink
project. We here follow a similar approach like the Stateful Functions effort,
where a separate repository has helped to speed up the development by allowing
for more light-weight contribution workflows and separate release cycles.

## Github organization created for Flink ecosystem projects

To facilitate the community collaboration on ecosystem projects that extend the
capability of the Apache Flink, Apache Flink PMC has granted the permission to
use flink-extended as the name of this [GitHub
organization](https://github.com/flink-extended), which provides a neutral
place to host the code of ecosystem projects.

Two Flink ML related projects have been moved to this organization.
[dl-on-flink](https://github.com/flink-extended/dl-on-flink) provides the
capability to implement Flink ML stages using TensorFlow. And
[clink](https://github.com/flink-extended/clink) is a library that facilitates
the implementation of Flink ML stages using C++ in order to support e.g.
real-time feature engineering.

We hope you can join this effort and share your Flink ecosystem projects in
this Github organization. And stay tuned for more updates on ecosystem
projects. 

# Upgrade Notes

Please review this note for a list of adjustments to make and issues to check
if you plan to upgrade to Flink ML 2.0.0.

This note discusses any critical information about incompatibilities and
breaking changes, performance changes, and any other changes that might impact
your production deployment of Flink ML.

* **Module names are changed**.

  We have replaced the `flink-ml-api` module with the `flink-ml-core_2.12`
  module.

  For users who have a dependency on `flink-ml-api`, please replace it with
  `flink-ml-core_2.12`
* **PipelineStage and its subclasses are changed**.

  [FLIP-173](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=184615783)
  made major changes to PipelineStage and its subclasses. Changes include class
  rename, method signature change, method removal etc.

  Users who use PipelineStage and its subclasses should use the new APIs
  introduced in FLIP-173.
* **Param-related classes are changed**.

  [FLIP-174](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=181311361)
  made major changes to the param-related classes. Changes include class rename,
  method signature change, method removal etc.

  Users who use classes such as Params and WithParams should use the new APIs
  introduced in FLIP-174.
* **Flink dependency is changed from 1.12 to 1.14**.

  This change introduces all the breaking changes listed in the Flink 1.14
  [release notes](https://nightlies.apache.org/flink/flink-docs-release-1.14/release-notes/flink-1.14).
  One major change is that the DataSet API is not supported anymore.

  Users who use DataSet::iterate should switch to using the datastream-based
  iteration API introduced in [FLIP-176](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=184615300).

# Release Notes and Resources

Please take a look at the [release notes](https://issues.apache.org/jira/secure/ReleaseNote.jspa?projectId=12315522&version=12351079)
for a detailed list of changes and new features.

The binary distribution and source artifacts are now available on the updated
[Downloads page](https://flink.apache.org/downloads.html) of the Flink website,
and the most recent distribution of Flink ML Python package is available on
[PyPI](https://pypi.org/project/apache-flink-ml).

# List of Contributors

The Apache Flink community would like to thank each one of the contributors
that have made this release possible:

Yun Gao, Dong Lin, Zhipeng Zhang, huangxingbo, Yunfeng Zhou, Jiangjie (Becket)
Qin, weibo, abdelrahman-ik.
