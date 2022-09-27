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

import org.apache.flink.api.common.time.Time;
import org.apache.flink.util.Preconditions;

/**
 * A windowing strategy that groups elements into sessions based on the current system time of the
 * machine the operation is running on. Windows do not overlap.
 */
public class ProcessingTimeSessionWindows implements Windows {
    /** The session timeout, i.e. the time gap between sessions. */
    private final Time gap;

    private ProcessingTimeSessionWindows(Time gap) {
        this.gap = Preconditions.checkNotNull(gap);
    }

    /**
     * Creates a new {@link ProcessingTimeSessionWindows}.
     *
     * @param gap The session timeout, i.e. the time gap between sessions
     */
    public static ProcessingTimeSessionWindows withGap(Time gap) {
        return new ProcessingTimeSessionWindows(gap);
    }

    public Time getGap() {
        return gap;
    }

    @Override
    public int hashCode() {
        return gap.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof ProcessingTimeSessionWindows)) {
            return false;
        }

        ProcessingTimeSessionWindows window = (ProcessingTimeSessionWindows) obj;

        return this.gap.equals(window.gap);
    }
}
