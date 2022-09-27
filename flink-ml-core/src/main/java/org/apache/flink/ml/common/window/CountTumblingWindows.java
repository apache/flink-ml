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

package org.apache.flink.ml.common.window;

import org.apache.flink.util.Preconditions;

/**
 * A windowing strategy that groups elements into fixed-size windows based on the count number of
 * the elements. Windows do not overlap.
 */
public class CountTumblingWindows implements Windows {
    /** Size of this window as row-count interval. */
    private final long size;

    private CountTumblingWindows(long size) {
        Preconditions.checkArgument(
                size > 0, "The size of a count window must be a positive value");
        this.size = size;
    }

    /**
     * Creates a new {@link CountTumblingWindows}.
     *
     * @param size the size of the window as row-count interval.
     */
    public static CountTumblingWindows of(long size) {
        return new CountTumblingWindows(size);
    }

    public long getSize() {
        return size;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(size);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof CountTumblingWindows)) {
            return false;
        }

        CountTumblingWindows windows = (CountTumblingWindows) obj;
        return this.size == windows.size;
    }
}
