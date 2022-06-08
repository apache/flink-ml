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

package org.apache.flink.iteration.config;

/** The strategy to store cache data. */
public enum DataCacheStrategy {
    ON_HEAP_AND_DISK(true, false),
    OFF_HEAP_AND_DISK(false, true),
    DISK_ONLY(false, false);

    /** Whether to use on-heap memory for cache. */
    public final boolean useOnHeap;

    /** Whether to use off-heap memory for cache. */
    public final boolean useOffHeap;

    DataCacheStrategy(boolean useOnHeap, boolean useOffHeap) {
        this.useOnHeap = useOnHeap;
        this.useOffHeap = useOffHeap;
    }
}
