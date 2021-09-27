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

package org.apache.flink.ml.iteration.operator.allround;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.ml.iteration.IterationRecord;
import org.apache.flink.ml.iteration.operator.OperatorWrapper;
import org.apache.flink.ml.iteration.proxy.ProxyKeySelector;
import org.apache.flink.ml.iteration.proxy.ProxyStreamPartitioner;
import org.apache.flink.ml.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.streaming.api.operators.MultipleInputStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.partitioner.StreamPartitioner;
import org.apache.flink.util.OutputTag;

/** The operator wrapper implementation for all-round wrappers. */
public class AllRoundOperatorWrapper<T> implements OperatorWrapper<T, IterationRecord<T>> {

    @Override
    public StreamOperator<IterationRecord<T>> wrap(
            StreamOperatorParameters<IterationRecord<T>> operatorParameters,
            StreamOperatorFactory<T> operatorFactory) {
        Class<? extends StreamOperator> operatorClass =
                operatorFactory.getStreamOperatorClass(getClass().getClassLoader());
        if (OneInputStreamOperator.class.isAssignableFrom(operatorClass)) {
            return new OneInputAllRoundWrapperOperator<>(operatorParameters, operatorFactory);
        } else if (TwoInputStreamOperator.class.isAssignableFrom(operatorClass)) {
            return new TwoInputAllRoundWrapperOperator<>(operatorParameters, operatorFactory);
        } else if (MultipleInputStreamOperator.class.isAssignableFrom(operatorClass)) {
            return new MultipleInputAllRoundWrapperOperator<>(operatorParameters, operatorFactory);
        } else {
            throw new UnsupportedOperationException(
                    "Unsupported operator class for all-round wrapper: " + operatorClass);
        }
    }

    @Override
    public <KEY> KeySelector<IterationRecord<T>, KEY> wrapKeySelector(
            KeySelector<T, KEY> keySelector) {
        return new ProxyKeySelector<>(keySelector);
    }

    @Override
    public StreamPartitioner<IterationRecord<T>> wrapStreamPartitioner(
            StreamPartitioner<T> streamPartitioner) {
        return new ProxyStreamPartitioner<>(streamPartitioner);
    }

    @Override
    public OutputTag<IterationRecord<T>> wrapOutputTag(OutputTag<T> outputTag) {
        return new OutputTag<>(
                outputTag.getId(), new IterationRecordTypeInfo<>(outputTag.getTypeInfo()));
    }

    @Override
    public TypeInformation<IterationRecord<T>> getWrappedTypeInfo(TypeInformation<T> typeInfo) {
        return new IterationRecordTypeInfo<>(typeInfo);
    }
}
