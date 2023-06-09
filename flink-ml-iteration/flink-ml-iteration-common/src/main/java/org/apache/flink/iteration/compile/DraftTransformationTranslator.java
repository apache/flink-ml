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

package org.apache.flink.iteration.compile;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.dag.Transformation;
import org.apache.flink.iteration.operator.OperatorWrapper;

/** Creates the actual transformation according to the draft transformation. */
public interface DraftTransformationTranslator<TF extends Transformation<?>> {

    Transformation<?> translate(
            TF draftTransformation, OperatorWrapper<?, ?> operatorWrapper, Context context);

    /** The context for {@link DraftTransformationTranslator}. */
    interface Context {

        Transformation<?> getActualTransformation(int draftId);

        ExecutionConfig getExecutionConfig();

        default Transformation<?> copyProperties(
                Transformation<?> actual, Transformation<?> draft) {
            actual.setName(draft.getName());
            actual.setParallelism(draft.getParallelism());

            if (draft.getMaxParallelism() > 0) {
                actual.setMaxParallelism(draft.getMaxParallelism());
            }

            if (draft.getBufferTimeout() > 0) {
                actual.setBufferTimeout(draft.getBufferTimeout());
            }

            if (draft.getSlotSharingGroup().isPresent()) {
                actual.setSlotSharingGroup(draft.getSlotSharingGroup().get());
            }
            actual.setCoLocationGroupKey(draft.getCoLocationGroupKey());

            actual.setUid(draft.getUid());
            if (draft.getUserProvidedNodeHash() != null) {
                actual.setUidHash(draft.getUserProvidedNodeHash());
            }

            draft.getManagedMemoryOperatorScopeUseCaseWeights()
                    .forEach(actual::declareManagedMemoryUseCaseAtOperatorScope);
            draft.getManagedMemorySlotScopeUseCases()
                    .forEach(actual::declareManagedMemoryUseCaseAtSlotScope);

            return actual;
        }
    }
}
