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

package org.apache.flink.ml.iteration.compile.translator;

import org.apache.flink.api.dag.Transformation;
import org.apache.flink.ml.iteration.compile.DraftTransformationTranslator;
import org.apache.flink.ml.iteration.operator.OperatorWrapper;
import org.apache.flink.streaming.api.transformations.PartitionTransformation;
import org.apache.flink.streaming.runtime.partitioner.StreamPartitioner;

/** Draft translator for the {@link PartitionTransformation}. */
public class PartitionTransformationTranslator
        implements DraftTransformationTranslator<PartitionTransformation<?>> {

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public Transformation<?> translate(
            PartitionTransformation<?> draftTransformation,
            OperatorWrapper<?, ?> operatorWrapper,
            Context context) {
        return context.copyProperties(
                new PartitionTransformation<>(
                        context.getActualTransformation(
                                draftTransformation.getInputs().get(0).getId()),
                        operatorWrapper.wrapStreamPartitioner(
                                (StreamPartitioner) draftTransformation.getPartitioner()),
                        draftTransformation.getShuffleMode()),
                draftTransformation);
    }
}
