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
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.ml.iteration.operator.OperatorWrapper;
import org.apache.flink.streaming.api.operators.MultipleInputStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.partitioner.StreamPartitioner;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

/** The operator wrapper for broadcast wrappers. */
public class BroadcastWrapper<T> implements OperatorWrapper<T, T> {
    /** name of the broadcast DataStreams. */
    private final String[] broadcastStreamNames;
    /** types of input DataStreams. */
    private final TypeInformation[] inTypes;
    /** whether each input is blocking or not. */
    private final boolean[] isBlocking;

    public BroadcastWrapper(String[] broadcastStreamNames, TypeInformation[] inTypes) {
        this(broadcastStreamNames, inTypes, new boolean[inTypes.length]);
    }

    public BroadcastWrapper(
            String[] broadcastStreamNames, TypeInformation[] inTypes, boolean[] isBlocking) {
        Preconditions.checkState(inTypes.length == isBlocking.length);
        this.broadcastStreamNames = broadcastStreamNames;
        this.inTypes = inTypes;
        this.isBlocking = isBlocking;
    }

    @Override
    public StreamOperator<T> wrap(
            StreamOperatorParameters<T> operatorParameters,
            StreamOperatorFactory<T> operatorFactory) {
        Class<? extends StreamOperator> operatorClass =
                operatorFactory.getStreamOperatorClass(getClass().getClassLoader());
        if (OneInputStreamOperator.class.isAssignableFrom(operatorClass)) {
            return new OneInputBroadcastWrapperOperator<>(
                    operatorParameters, operatorFactory, broadcastStreamNames, inTypes, isBlocking);
        } else if (TwoInputStreamOperator.class.isAssignableFrom(operatorClass)) {
            return new TwoInputBroadcastWrapperOperator<>(
                    operatorParameters, operatorFactory, broadcastStreamNames, inTypes, isBlocking);
        } else if (MultipleInputStreamOperator.class.isAssignableFrom(operatorClass)) {
            return new MultipleInputBroadcastWrapperOperator<>(
                    operatorParameters, operatorFactory, broadcastStreamNames, inTypes, isBlocking);
        } else {
            throw new UnsupportedOperationException(
                    "Unsupported operator class for with-broadcast wrapper: " + operatorClass);
        }
    }

    @Override
    public <KEY> KeySelector<T, KEY> wrapKeySelector(KeySelector<T, KEY> keySelector) {
        return keySelector;
    }

    @Override
    public StreamPartitioner<T> wrapStreamPartitioner(StreamPartitioner<T> streamPartitioner) {
        return streamPartitioner;
    }

    @Override
    public OutputTag<T> wrapOutputTag(OutputTag<T> outputTag) {
        return outputTag;
    }

    @Override
    public TypeInformation<T> getWrappedTypeInfo(TypeInformation<T> typeInfo) {
        return typeInfo;
    }
}
