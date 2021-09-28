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

package org.apache.flink.ml.iteration.broadcast;

import org.apache.flink.metrics.Counter;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import java.io.IOException;
import java.util.List;

/**
 * The intermediate broadcast output that wrappers a list of internal outputs. It will broadcast the
 * records to all the internal outputs and increment the counter.
 */
public class CountingBroadcastOutput<OUT> implements BroadcastOutput<OUT> {

    private final Counter numRecordsOut;
    private final List<BroadcastOutput<OUT>> internalOutputs;

    public CountingBroadcastOutput(
            Counter numRecordsOut, List<BroadcastOutput<OUT>> internalOutputs) {
        this.numRecordsOut = numRecordsOut;
        this.internalOutputs = internalOutputs;
    }

    @Override
    public void broadcastEmit(StreamRecord<OUT> record) throws IOException {
        numRecordsOut.inc();

        for (BroadcastOutput<OUT> internalOutput : internalOutputs) {
            internalOutput.broadcastEmit(record);
        }
    }
}
