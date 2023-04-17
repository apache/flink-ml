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

package org.apache.flink.iteration.operator;

import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.TestLogger;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

/** Tests the {@link OperatorUtils}. */
public class OperatorUtilsTest extends TestLogger {

    @Test
    public void testCreateWrappedOperatorConfig() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        env.getConfig().enableObjectReuse();

        final OutputTag<Integer> outputTag = new OutputTag("0", Types.INT) {};
        final Integer[] sourceData = new Integer[] {1, 2, 3};

        DataStream<Integer> variableStream =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {});
        DataStream<Integer> dataStream = env.fromElements(sourceData);

        DataStreamList result =
                Iterations.iterateUnboundedStreams(
                        DataStreamList.of(variableStream),
                        DataStreamList.of(dataStream),
                        (variableStreams, dataStreams) -> {
                            SingleOutputStreamOperator transformed =
                                    dataStreams
                                            .<Integer>get(0)
                                            .transform(
                                                    "side-output",
                                                    Types.INT,
                                                    new SideOutputOperator(outputTag));
                            return new IterationBodyResult(
                                    DataStreamList.of(variableStreams.get(0)),
                                    DataStreamList.of(transformed.getSideOutput(outputTag)));
                        });
        assertEquals(
                Arrays.asList(sourceData), IteratorUtils.toList(result.get(0).executeAndCollect()));
    }

    private static class SideOutputOperator extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<Integer, Integer> {

        private final OutputTag<Integer> outputTag;

        public SideOutputOperator(OutputTag<Integer> outputTag) {
            this.outputTag = outputTag;
        }

        @Override
        public void open() throws Exception {
            super.open();
            StreamConfig config = getOperatorConfig();
            ClassLoader cl = getClass().getClassLoader();

            assertEquals(IntSerializer.INSTANCE, config.getTypeSerializerIn(0, cl));
            assertEquals(IntSerializer.INSTANCE, config.getTypeSerializerOut(cl));
            assertEquals(IntSerializer.INSTANCE, config.getTypeSerializerSideOut(outputTag, cl));
        }

        @Override
        public void processElement(StreamRecord<Integer> element) {
            output.collect(outputTag, element);
        }
    }
}
