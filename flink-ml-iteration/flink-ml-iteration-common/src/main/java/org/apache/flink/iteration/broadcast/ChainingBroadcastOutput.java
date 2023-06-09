/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.iteration.broadcast;

import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.OutputTag;

/** The broadcast output corresponding to a chained output. */
public class ChainingBroadcastOutput<OUT> implements BroadcastOutput<OUT> {
    private final Output<StreamRecord<OUT>> rawOutput;
    private final OutputTag outputTag;

    ChainingBroadcastOutput(Output<StreamRecord<OUT>> rawOutput, OutputTag outputTag) {
        this.rawOutput = rawOutput;
        this.outputTag = outputTag;
    }

    @SuppressWarnings("unchecked")
    @Override
    public void broadcastEmit(StreamRecord<OUT> record) {
        if (outputTag == null) {
            rawOutput.collect(record);
        } else {
            rawOutput.collect(outputTag, record);
        }
    }
}
