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

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.metrics.Counter;
import org.apache.flink.runtime.io.network.api.writer.RecordWriter;
import org.apache.flink.runtime.plugable.SerializationDelegate;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.runtime.streamrecord.StreamElement;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.OutputTag;

import java.util.ArrayList;
import java.util.List;

/** Factory that creates the corresponding {@link BroadcastOutput} from the {@link Output}. */
public class BroadcastOutputFactory {

    /**
     * Creates the wrapper broadcast output from {@code output}.
     *
     * @param output the original output.
     * @param numRecordsOut the counter for the number of output record.
     * @return the wrapped broadcast output.
     */
    public static <OUT> BroadcastOutput<OUT> createBroadcastOutput(
            Output<StreamRecord<OUT>> output, Counter numRecordsOut) {

        OutputReflectionContext outputReflectionContext = new OutputReflectionContext();

        if (outputReflectionContext.isCountingOutput(output)) {
            output = outputReflectionContext.getCountingInternalOutput(output);
        }

        List<BroadcastOutput<OUT>> internalOutputs = new ArrayList<>();
        if (outputReflectionContext.isBroadcastingOutput(output)) {
            Output<StreamRecord<OUT>>[] rawOutputs =
                    outputReflectionContext.getBroadcastingInternalOutputs(output);
            for (Output<StreamRecord<OUT>> rawOutput : rawOutputs) {
                internalOutputs.add(
                        createInternalBroadcastOutput(rawOutput, outputReflectionContext));
            }
        } else {
            internalOutputs.add(createInternalBroadcastOutput(output, outputReflectionContext));
        }

        return new CountingBroadcastOutput<>(numRecordsOut, internalOutputs);
    }

    private static <OUT> BroadcastOutput<OUT> createInternalBroadcastOutput(
            Output<StreamRecord<OUT>> rawOutput, OutputReflectionContext outputReflectionContext) {

        if (outputReflectionContext.isChainingOutput(rawOutput)) {
            OutputTag<?> outputTag = outputReflectionContext.getChainingOutputTag(rawOutput);
            return new ChainingBroadcastOutput<>(rawOutput, outputTag);
        } else if (outputReflectionContext.isRecordWriterOutput(rawOutput)) {
            RecordWriter<SerializationDelegate<StreamElement>> recordWriter =
                    outputReflectionContext.getRecordWriter(rawOutput);
            TypeSerializer<StreamElement> typeSerializer =
                    outputReflectionContext.getRecordWriterTypeSerializer(rawOutput);
            return new RecordWriterBroadcastOutput<>(recordWriter, typeSerializer);
        } else {
            throw new RuntimeException("Unknown output type: " + rawOutput.getClass());
        }
    }
}
