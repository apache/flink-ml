/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.common.ps;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.common.ps.message.AllReduceM;
import org.apache.flink.ml.common.ps.message.InitializeModelAsZeroM;
import org.apache.flink.ml.common.ps.message.PullIndexM;
import org.apache.flink.ml.common.ps.message.PushKvM;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import java.util.Arrays;
import java.util.Iterator;
import java.util.function.BiFunction;

/** ServerAgent resides on each worker. It serves as an agent for workers to talk with servers. */
public class ServerAgent {
    /** Index of the worker that this agent resides on. */
    private final int workerId;
    /** Partitioner of the model data that this ServerAgent maintains. */
    private RangePartitioner partitioner;
    /** The collector on this worker. */
    private final Output<StreamRecord<Tuple2<Integer, byte[]>>> output;

    ServerAgent(int workerId, Output<StreamRecord<Tuple2<Integer, byte[]>>> output) {
        this.workerId = workerId;
        this.output = output;
    }

    void setPartitioner(RangePartitioner partitioner) {
        this.partitioner = partitioner;
    }

    /** Sends a request to servers to initialize the values stored as zeros. */
    void initializeModelAsZeros() {
        for (int serverId = 0; serverId < partitioner.numServers; serverId++) {
            long start = partitioner.ranges[serverId];
            long end = partitioner.ranges[serverId + 1];
            InitializeModelAsZeroM initializeModelAsZeroM =
                    new InitializeModelAsZeroM(workerId, serverId, start, end);
            output.collect(
                    new StreamRecord<>(Tuple2.of(serverId, initializeModelAsZeroM.toBytes())));
        }
    }

    /** Pushes a key-value arrays to servers. */
    void push(long[] indices, double[] values) {
        Iterator<Tuple3<Integer, long[], double[]>> requests =
                partitioner.splitRequest(indices, values);
        while (requests.hasNext()) {
            Tuple3<Integer, long[], double[]> request = requests.next();
            PushKvM pushKvM = new PushKvM(workerId, request.f0, Tuple2.of(request.f1, request.f2));
            output.collect(new StreamRecord<>(Tuple2.of(request.f0, pushKvM.toBytes())));
        }
    }

    /** Pulls the values from servers with the specified indices. */
    void pull(long[] indices) {
        Iterator<Tuple3<Integer, long[], double[]>> requests =
                partitioner.splitRequest(indices, null);
        while (requests.hasNext()) {
            Tuple3<Integer, long[], double[]> request = requests.next();
            PullIndexM pullIndexM = new PullIndexM(request.f0, workerId, request.f1);
            output.collect(new StreamRecord<>(Tuple2.of(request.f0, pullIndexM.toBytes())));
        }
    }

    /**
     * Pushes the values to servers to apply all reduce operation.
     *
     * <p>Note that the values pushed by this function are not going to update the model, but just
     * perform an all reduce operation.
     */
    void allReducePush(double[] values, BiFunction<double[], double[], double[]> aggregator) {
        final int MIN_MESSAGE_SIZE = 1024;
        int numServers = partitioner.numServers;
        int messageSize = Math.max(MIN_MESSAGE_SIZE, values.length / numServers + 1);
        for (int serverId = 0; serverId < numServers; serverId++) {
            int s = Math.min(serverId * messageSize, values.length);
            int e = Math.min(s + messageSize, values.length);
            double[] segment;
            if (s == e) {
                segment = new double[0];
            } else {
                segment = Arrays.copyOfRange(values, s, e);
            }
            AllReduceM allReduceM = new AllReduceM(serverId, workerId, segment, aggregator);
            output.collect(new StreamRecord<>(Tuple2.of(serverId, allReduceM.toBytes())));
        }
    }
}
