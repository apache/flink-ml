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

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.accumulators.Accumulator;
import org.apache.flink.api.common.functions.BroadcastVariableInitializer;
import org.apache.flink.api.common.state.KeyedStateStore;
import org.apache.flink.metrics.groups.OperatorMetricGroup;
import org.apache.flink.runtime.execution.Environment;
import org.apache.flink.runtime.externalresource.ExternalResourceInfoProvider;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.streaming.api.operators.StreamingRuntimeContext;
import org.apache.flink.streaming.runtime.tasks.ProcessingTimeService;

import javax.annotation.Nullable;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An subclass of {@link StreamingRuntimeContext} that provides accessibility of broadcast
 * variables.
 */
public class BroadcastStreamingRuntimeContext extends StreamingRuntimeContext {

    Map<String, List<?>> broadcastVariables = new HashMap<>();

    public BroadcastStreamingRuntimeContext(
            Environment env,
            Map<String, Accumulator<?, ?>> accumulators,
            OperatorMetricGroup operatorMetricGroup,
            OperatorID operatorID,
            ProcessingTimeService processingTimeService,
            @Nullable KeyedStateStore keyedStateStore,
            ExternalResourceInfoProvider externalResourceInfoProvider) {
        super(
                env,
                accumulators,
                operatorMetricGroup,
                operatorID,
                processingTimeService,
                keyedStateStore,
                externalResourceInfoProvider);
    }

    @Override
    public boolean hasBroadcastVariable(String name) {
        return broadcastVariables.containsKey(name);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <RT> List<RT> getBroadcastVariable(String name) {
        if (broadcastVariables.containsKey(name)) {
            return (List<RT>) broadcastVariables.get(name);
        } else {
            throw new RuntimeException(
                    "Cannot get broadcast variables before processing elements.");
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T, C> C getBroadcastVariableWithInitializer(
            String name, BroadcastVariableInitializer<T, C> initializer) {
        if (broadcastVariables.containsKey(name)) {
            return initializer.initializeBroadcastVariable((List<T>) broadcastVariables.get(name));
        } else {
            throw new RuntimeException(
                    "Cannot get broadcast variables before processing elements.");
        }
    }

    @Internal
    public void setBroadcastVariable(String name, List<?> broadcastVariable) {
        broadcastVariables.put(name, broadcastVariable);
    }
}
