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

package org.apache.flink.iteration.datacache.nonkeyed;

import org.apache.flink.annotation.Internal;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.MemorySize;
import org.apache.flink.configuration.TaskManagerOptions;
import org.apache.flink.iteration.config.IterationOptions;
import org.apache.flink.util.Preconditions;

/** A class that manages bookkeeping for a fixed-size heap memory space. */
@Internal
public class OnHeapMemoryPool {

    private static Configuration configuration;

    private static OnHeapMemoryPool onHeapMemoryPool;

    /** The number of bytes this pool can allocate at most. */
    private final long poolSize;

    /** The number of allocated bytes. */
    private long memoryUsed;

    OnHeapMemoryPool(long poolSize) {
        this.poolSize = poolSize;
        this.memoryUsed = 0L;
    }

    /**
     * Gets or creates the {@link OnHeapMemoryPool} instance from the provided {@param
     * configuration} and registers it as a singleton object. There would be only one {@link
     * OnHeapMemoryPool} instance for each JVM (TaskManager).
     */
    public static OnHeapMemoryPool getOrCreate(Configuration configuration) {
        if (onHeapMemoryPool != null) {
            Preconditions.checkArgument(OnHeapMemoryPool.configuration.equals(configuration));
            return onHeapMemoryPool;
        }

        OnHeapMemoryPool.configuration = configuration;
        double dataCacheHeapMemoryFraction =
                configuration.get(IterationOptions.DATA_CACHE_HEAP_MEMORY_FRACTION);
        MemorySize memorySize =
                configuration
                        .get(TaskManagerOptions.TASK_HEAP_MEMORY)
                        .multiply(dataCacheHeapMemoryFraction);

        onHeapMemoryPool = new OnHeapMemoryPool(memorySize.getBytes());
        return onHeapMemoryPool;
    }

    long getMemoryUsed() {
        return memoryUsed;
    }

    /**
     * Acquires a certain number of bytes from the memory space.
     *
     * @return true if the bytes are successfully acquired, false if there is not enough space for
     *     the bytes.
     */
    public boolean acquireMemory(long numBytesToAcquire) {
        if (numBytesToAcquire + memoryUsed > poolSize) {
            return false;
        }
        memoryUsed += numBytesToAcquire;
        return true;
    }

    /** Returns a certain number of bytes to the memory space. */
    public void releaseMemory(long numBytesToRelease) {
        memoryUsed -= numBytesToRelease;
    }
}
