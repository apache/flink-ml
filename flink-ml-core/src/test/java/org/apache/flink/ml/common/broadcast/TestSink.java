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

package org.apache.flink.ml.common.broadcast;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.common.broadcast.operator.BroadcastVariableReceiverOperatorTest;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;

import org.apache.commons.collections.IteratorUtils;

import java.util.List;

/**
 * Utility class to check the result of a sink. It throws an exception if the records received are
 * not as expected.
 */
public class TestSink extends RichSinkFunction<Integer> implements CheckpointedFunction {

    private final List<Integer> expectReceivedRecords;

    private List<Integer> receivedRecords;

    private ListState<Integer> receivedRecordsState;

    public TestSink(List<Integer> expectReceivedRecords) {
        this.expectReceivedRecords = expectReceivedRecords;
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
    }

    @Override
    public void invoke(Integer value, Context context) {
        receivedRecords.add(value);
    }

    @Override
    public void finish() {
        BroadcastVariableReceiverOperatorTest.compareLists(expectReceivedRecords, receivedRecords);
    }

    @Override
    public void close() {}

    @Override
    public void snapshotState(FunctionSnapshotContext functionSnapshotContext) throws Exception {
        this.receivedRecordsState.clear();
        this.receivedRecordsState.addAll(receivedRecords);
    }

    @Override
    public void initializeState(FunctionInitializationContext functionInitializationContext)
            throws Exception {
        receivedRecordsState =
                functionInitializationContext
                        .getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "receivedRecords", BasicTypeInfo.INT_TYPE_INFO));
        receivedRecords = IteratorUtils.toList(receivedRecordsState.get().iterator());
    }
}
