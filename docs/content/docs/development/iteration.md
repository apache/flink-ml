---
title: "Iteration"
weight: 2
type: docs
aliases:
- /development/iteration.html
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

# Iteration

Iteration is a basic building block for a ML library. In machine learning
algorithms, iteration might be used in offline or online training process. In
general, two types of iterations are required and Flink ML supports both of them
in order to provide the infrastructure for a variety of algorithms.

1. **Bounded Iteration**: Usually used in the offline case. In this case, the
   algorithm usually trains on a bounded dataset. It updates the parameters for
   multiple rounds until convergence.
2. **Unbounded Iteration**: Usually used in the online case, in this case, the
   algorithm usually trains on an unbounded dataset. It accumulates a mini-batch
   of data and then do one update to the parameters. 

## Iteration Paradigm

An iterative algorithm has the following behavior pattern:

- The iterative algorithm has an ***iteration body*** that is repeatedly invoked
  until some termination criteria is reached (e.g. after a user-specified number
  of epochs has been reached). An iteration body is a subgraph of operators that
  implements the computation logic of e.g. an iterative machine learning
  algorithm, whose outputs might be fed back as the inputs of this subgraph. 
- In each invocation, the iteration body updates the model parameters based on
  the user-provided data as well as the most recent model parameters.
- The iterative algorithm takes as inputs the user-provided data and the initial
  model parameters.
- The iterative algorithm could output arbitrary user-defined information, such
  as the loss after each epoch, or the final model parameters. 

Therefore, the behavior of an iterative algorithm could be characterized with
the following iteration paradigm (w.r.t. Flink concepts):

- An iteration-body is a Flink subgraph with the following inputs and outputs:
  - Inputs: **model-variables** (as a list of data streams) and
    **user-provided-data** (as another list of data streams)
  - Outputs: **feedback-model-variables** (as a list of data streams) and
    **user-observed-outputs** (as a list of data streams)
- A **termination-condition** that specifies when the iterative execution of the
  iteration body should terminate.
- In order to execute an iteration body, a user needs to execute the iteration
  body with the following inputs, and gets the following outputs.
  - Inputs: **initial-model-variables** (as a list of bounded data streams) and
    **user-provided-data** (as a list of data streams)
  - Outputs: the **user-observed-output** emitted by the iteration body.

It is important to note that the **model-variables** expected by the iteration
body is not the same as the **initial-model-variables** provided by the user.
Instead, **model-variables** are computed as the union of the
**feedback-model-variables** (emitted by the iteration body) and the
**initial-model-variables** (provided by the caller of the iteration body).
Flink ML provides utility class (see Iterations) to run an iteration-body with
the user-provided inputs.

The figure below summarizes the iteration paradigm described above. 

{{<  mermaid >}}
flowchart LR

subgraph Iteration Body
union1
union2
node11
node12
node21
node22
nodeX
end

input0 --> node11

union1 -. feedback .-  node12
input1 --> union1
union1 --> node11
node11 --> nodeX
nodeX --> node12
node12 --> output1

input2 --> union2
union2 --> node21
node21 --> nodeX
nodeX --> node22
node22 --> output2
union2 -. feedback .-  node22

input0[non-iterate input]
input1[iterate input]
input2[iterate input]
union1[union]
union2[union]
node11( )
node12( )
nodeX( )
node21( )
node22( )
output1[output]
output2[output]

{{<  /mermaid >}}

## API

The main entry of Flink ML's iteration lies in `Iterations` class. It mainly
provides two public methods and users may choose to use either of them based on
whether the input data is bounded or unbounded.

```java
public class Iterations {
  public static DataStreamList iterateUnboundedStreams(
    DataStreamList initVariableStreams, DataStreamList dataStreams, IterationBody body) {...}
  ...
  public static DataStreamList iterateBoundedStreamsUntilTermination(
    DataStreamList initVariableStreams,
    ReplayableDataStreamList dataStreams,
    IterationConfig config,
    IterationBody body){...}
}
```

To construct an iteration, Users are required to provide

- `initVariableStreams`: the initial values of the variable data streams which
  would be updated in each round.
- `dataStreams`: the other data streams used inside the iteration, but would not
  be updated.
- `iterationBody`: specifies the subgraph to update the variable streams and the
  outputs.

The `IterationBody` will be invoked with two parameters: The first parameter is
a list of input variable streams, which are created as the union of the initial
variable streams and the corresponding feedback variable streams (returned by
the iteration body); The second parameter is the data streams given to this
method. 

```java
public interface IterationBody extends Serializable {
  ...
  IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams);
  ...
}
```

During the execution of iteration body, each of the records involved in the
iteration has an epoch attached, which marks the progress of the iteration. The
epoch is computed as:

- All records in the initial variable streams and initial data streams has epoch
  = 0.
- For any record emitted by this operator into a non-feedback stream, the epoch
  of this emitted record = the epoch of the input record that triggers this
  emission. If this record is emitted by onEpochWatermarkIncremented(), then the
  epoch of this record = epochWatermark.
- For any record emitted by this operator into a feedback variable stream, the
  epoch of the emitted record = the epoch of the input record that triggers this
  emission + 1.

The framework would deliver notification at the end of each epoch to operators
and UDFs that implements `IterationListener`.

```java
public interface IterationListener<T> {
  void onEpochWatermarkIncremented(int epochWatermark, Context context, Collector<T> collector)
    throws Exception;
  ...
  void onIterationTerminated(Context context, Collector<T> collector) throws Exception;
}
```

## Example Usage

Example codes of utilizing iterations is as below。

```java
DataStream<double[]> initParameters = ... 
DataStream<Tuple2<double[], Double>> dataset = ...

DataStreamList resultStreams = Iterations.iterateUnboundedStreams(
	DataStreamList.of(initParameters),
  ReplayableDataStreamList.notReplay(dataset),
  IterationConfig.newBuilder().setOperatorRoundMode(ALL_ROUND).build();
  (variableStreams, dataStreams) -> {
    DataStream<double[]> modelUpdate = variableStreams.get(0); 
    DataStream<Tuple2<double[], Double>> dataset = dataStreams.get(0);
    DataStream<double[]> newModelUpdate = ... 
    DataStream<double[]> modelOutput = ... 
    return new IterationBodyResult(
      DataStreamList.of(newModelUpdate), 
      DataStreamList.of(modelOutput)
});

DataStream<double[]> finalModel = resultStreams.get("final_model");
```

- `initParameters`: input data that needs to be transmitted through feedback
  edge.
- `dataset`: input data that does not need to be transmitted through feedback
  edge.
- `newModelUpdate`: data to be transmitted through feedback edge
- `modelOutput`: final output of the iteration body
