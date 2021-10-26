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

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.runtime.partitioner.StreamPartitioner;
import org.apache.flink.util.OutputTag;

import java.io.Serializable;

/** Wrappers for the given operator factory. */
public interface OperatorWrapper<T, R> extends Serializable {

    StreamOperator<R> wrap(
            StreamOperatorParameters<R> operatorParameters,
            StreamOperatorFactory<T> operatorFactory);

    <KEY> KeySelector<R, KEY> wrapKeySelector(KeySelector<T, KEY> keySelector);

    StreamPartitioner<R> wrapStreamPartitioner(StreamPartitioner<T> streamPartitioner);

    OutputTag<R> wrapOutputTag(OutputTag<T> outputTag);

    TypeInformation<R> getWrappedTypeInfo(TypeInformation<T> typeInfo);
}
