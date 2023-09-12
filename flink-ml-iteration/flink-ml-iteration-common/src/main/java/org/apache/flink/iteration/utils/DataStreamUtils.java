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

package org.apache.flink.iteration.utils;

import org.apache.flink.core.memory.ManagedMemoryUseCase;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.TableException;

import java.util.Optional;

/** Provides utility functions for {@link DataStream}. */
public class DataStreamUtils {

    /**
     * Sets {Transformation#declareManagedMemoryUseCaseAtOperatorScope(ManagedMemoryUseCase, int)}
     * using the given bytes for {@link ManagedMemoryUseCase#OPERATOR}.
     *
     * <p>This method is in reference to Flink's ExecNodeUtil.setManagedMemoryWeight. The provided
     * bytes should be in the same scale as existing usage in Flink, for example,
     * StreamExecWindowAggregate.WINDOW_AGG_MEMORY_RATIO.
     */
    public static <T> void setManagedMemoryWeight(DataStream<T> dataStream, long memoryBytes) {
        if (memoryBytes > 0) {
            final int weightInMebibyte = Math.max(1, (int) (memoryBytes >> 20));
            final Optional<Integer> previousWeight =
                    dataStream
                            .getTransformation()
                            .declareManagedMemoryUseCaseAtOperatorScope(
                                    ManagedMemoryUseCase.OPERATOR, weightInMebibyte);
            if (previousWeight.isPresent()) {
                throw new TableException(
                        "Managed memory weight has been set, this should not happen.");
            }
        }
    }
}
