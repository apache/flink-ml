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

package org.apache.flink.ml.common.sharedobjects;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.iteration.operator.OperatorWrapper;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorFactoryUtil;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.partitioner.StreamPartitioner;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.util.OutputTag;

/** The operator wrapper for {@link AbstractSharedObjectsWrapperOperator}. */
class SharedObjectsWrapper<T> implements OperatorWrapper<T, T> {

    /** Shared objects context. */
    private final SharedObjectsContextImpl context;

    public SharedObjectsWrapper(SharedObjectsContextImpl context) {
        this.context = context;
    }

    @Override
    public StreamOperator<T> wrap(
            StreamOperatorParameters<T> operatorParameters,
            StreamOperatorFactory<T> operatorFactory) {
        Class<? extends StreamOperator> operatorClass =
                operatorFactory.getStreamOperatorClass(getClass().getClassLoader());
        if (SharedObjectsStreamOperator.class.isAssignableFrom(operatorClass)) {
            if (OneInputStreamOperator.class.isAssignableFrom(operatorClass)) {
                return new OneInputSharedObjectsWrapperOperator<>(
                        operatorParameters, operatorFactory, context);
            } else if (TwoInputStreamOperator.class.isAssignableFrom(operatorClass)) {
                return new TwoInputSharedObjectsWrapperOperator<>(
                        operatorParameters, operatorFactory, context);
            } else {
                return nowrap(operatorParameters, operatorFactory);
            }
        } else {
            return nowrap(operatorParameters, operatorFactory);
        }
    }

    public StreamOperator<T> nowrap(
            StreamOperatorParameters<T> parameters, StreamOperatorFactory<T> operatorFactory) {
        return StreamOperatorFactoryUtil.createOperator(
                        operatorFactory,
                        (StreamTask<T, ?>) parameters.getContainingTask(),
                        parameters.getStreamConfig(),
                        parameters.getOutput(),
                        parameters.getOperatorEventDispatcher())
                .f0;
    }

    @Override
    public Class<? extends StreamOperator> getStreamOperatorClass(
            ClassLoader classLoader, StreamOperatorFactory<T> operatorFactory) {
        Class<? extends StreamOperator> operatorClass =
                operatorFactory.getStreamOperatorClass(getClass().getClassLoader());
        if (OneInputStreamOperator.class.isAssignableFrom(operatorClass)) {
            return OneInputSharedObjectsWrapperOperator.class;
        } else if (TwoInputStreamOperator.class.isAssignableFrom(operatorClass)) {
            return TwoInputSharedObjectsWrapperOperator.class;
        } else {
            throw new UnsupportedOperationException(
                    "Unsupported operator class for shared objects wrapper: " + operatorClass);
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
