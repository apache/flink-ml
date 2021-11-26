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

package org.apache.flink.ml.common.broadcast.operator;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;

import java.io.Serializable;

/** Factory class for {@link BroadcastVariableReceiverOperator}. */
public class BroadcastVariableReceiverOperatorFactory<OUT>
        extends AbstractStreamOperatorFactory<OUT> implements Serializable {

    /** names of the broadcast data streams. */
    private final String[] broadcastNames;

    /** types of the broadcast data streams. */
    private final TypeInformation<?>[] inTypes;

    public BroadcastVariableReceiverOperatorFactory(
            String[] broadcastNames, TypeInformation<?>[] inTypes) {
        this.broadcastNames = broadcastNames;
        this.inTypes = inTypes;
    }

    @Override
    public <T extends StreamOperator<OUT>> T createStreamOperator(
            StreamOperatorParameters<OUT> parameters) {
        return (T) new BroadcastVariableReceiverOperator(parameters, broadcastNames, inTypes);
    }

    @Override
    public Class<? extends StreamOperator> getStreamOperatorClass(ClassLoader classLoader) {
        return BroadcastVariableReceiverOperator.class;
    }
}
