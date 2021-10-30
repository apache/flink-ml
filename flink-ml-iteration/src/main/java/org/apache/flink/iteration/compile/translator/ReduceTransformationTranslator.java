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

package org.apache.flink.iteration.compile.translator;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.dag.Transformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.iteration.compile.DraftTransformationTranslator;
import org.apache.flink.iteration.operator.OperatorWrapper;
import org.apache.flink.iteration.operator.WrapperOperatorFactory;
import org.apache.flink.streaming.api.operators.SimpleOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamGroupedReduceOperator;
import org.apache.flink.streaming.api.transformations.OneInputTransformation;
import org.apache.flink.streaming.api.transformations.ReduceTransformation;

/** Draft translator for the {@link ReduceTransformation}. */
public class ReduceTransformationTranslator
        implements DraftTransformationTranslator<ReduceTransformation<?, ?>> {

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public Transformation<?> translate(
            ReduceTransformation<?, ?> draftTransformation,
            OperatorWrapper<?, ?> operatorWrapper,
            Context context) {

        StreamGroupedReduceOperator<?> groupedReduce =
                new StreamGroupedReduceOperator(
                        draftTransformation.getReducer(),
                        draftTransformation
                                .getInputType()
                                .createSerializer(context.getExecutionConfig()));
        OneInputTransformation<?, ?> actualTransformation =
                new OneInputTransformation(
                        context.getActualTransformation(
                                draftTransformation.getInputs().get(0).getId()),
                        draftTransformation.getName(),
                        new WrapperOperatorFactory(
                                SimpleOperatorFactory.of(groupedReduce), operatorWrapper),
                        operatorWrapper.getWrappedTypeInfo(
                                (TypeInformation) draftTransformation.getOutputType()),
                        draftTransformation.getParallelism());

        actualTransformation.setStateKeyType(draftTransformation.getKeyTypeInfo());
        actualTransformation.setStateKeySelector(
                operatorWrapper.wrapKeySelector(
                        (KeySelector) draftTransformation.getKeySelector()));

        actualTransformation.setChainingStrategy(draftTransformation.getChainingStrategy());
        return context.copyProperties(actualTransformation, draftTransformation);
    }
}
