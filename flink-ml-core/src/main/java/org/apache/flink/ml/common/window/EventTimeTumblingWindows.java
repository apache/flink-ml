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
 * A windowing strategy that groups elements into fixed-size windows based on the timestamp of the
 * elements. Windows do not overlap.
 */
public class EventTimeTumblingWindows implements Windows {
    /** Size of this window as time interval. */
    private final Time size;

    private EventTimeTumblingWindows(Time size) {
        this.size = Preconditions.checkNotNull(size);
    }

    /**
     * Creates a new {@link EventTimeTumblingWindows}.
     *
     * @param size the size of the window as time interval.
     */
    public static EventTimeTumblingWindows of(Time size) {
        return new EventTimeTumblingWindows(size);
    }

    public Time getSize() {
        return size;
    }

    @Override
    public int hashCode() {
        return size.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof EventTimeTumblingWindows)) {
            return false;
        }

        EventTimeTumblingWindows window = (EventTimeTumblingWindows) obj;

        return this.size.equals(window.size);
    }
}
