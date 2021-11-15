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

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;

/** The factory to create the subclass of {@link AbstractWrapperOperator}. */
public class WrapperOperatorFactory<OUT>
        extends AbstractStreamOperatorFactory<IterationRecord<OUT>> {

    private final StreamOperatorFactory<OUT> operatorFactory;

    private final OperatorWrapper<OUT, IterationRecord<OUT>> wrapper;

    public WrapperOperatorFactory(
            StreamOperatorFactory<OUT> operatorFactory,
            OperatorWrapper<OUT, IterationRecord<OUT>> wrapper) {
        this.operatorFactory = operatorFactory;
        this.wrapper = wrapper;
    }

    @Override
    public <T extends StreamOperator<IterationRecord<OUT>>> T createStreamOperator(
            StreamOperatorParameters<IterationRecord<OUT>> parameters) {
        return (T) wrapper.wrap(parameters, operatorFactory);
    }

    @Override
    public Class<? extends StreamOperator> getStreamOperatorClass(ClassLoader classLoader) {
        return wrapper.getStreamOperatorClass(classLoader, operatorFactory);
    }

    @VisibleForTesting
    public StreamOperatorFactory<OUT> getOperatorFactory() {
        return operatorFactory;
    }

    @VisibleForTesting
    public OperatorWrapper<OUT, IterationRecord<OUT>> getWrapper() {
        return wrapper;
    }
}
