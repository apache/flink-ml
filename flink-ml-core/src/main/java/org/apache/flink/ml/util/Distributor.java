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

package org.apache.flink.ml.util;

import java.io.Serializable;

/**
 * A utility class which helps data partitioning.
 *
 * <p>Given an indexable linear structures, like an array, of n elements and m tasks, the goal is to
 * partition the linear structure into m consecutive segments and assign them to tasks accordingly.
 * This class calculates the segment assigned to each task, including the start position and element
 * count of the segment.
 */
public abstract class Distributor implements Serializable {
    protected final long numTasks;
    protected final long total;

    public Distributor(long total, long numTasks) {
        this.numTasks = numTasks;
        this.total = total;
    }

    /**
     * Calculates the start position of the segment assigned to the task.
     *
     * @param taskId The task index.
     * @return The start position.
     */
    public abstract long start(long taskId);

    /**
     * Calculates the count of elements of the segment assigned to the task.
     *
     * @param taskId The task index.
     * @return The count of elements.
     */
    public abstract long count(long taskId);

    /** An implementation of {@link Distributor} which evenly partitioned the elements. */
    public static class EvenDistributor extends Distributor {

        public EvenDistributor(long parallelism, long totalCnt) {
            super(totalCnt, parallelism);
        }

        @Override
        public long start(long taskId) {
            long div = total / numTasks;
            long mod = total % numTasks;
            return taskId < mod ? div * taskId + taskId : div * taskId + mod;
        }

        @Override
        public long count(long taskId) {
            long div = total / numTasks;
            long mod = total % numTasks;
            return taskId < mod ? div + 1 : div;
        }
    }
}
