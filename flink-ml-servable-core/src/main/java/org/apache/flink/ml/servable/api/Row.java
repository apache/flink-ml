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

package org.apache.flink.ml.servable.api;

import org.apache.flink.annotation.PublicEvolving;

import java.util.List;

/** Represents an ordered list of values. */
@PublicEvolving
public class Row {
    private final List<Object> values;

    /**
     * The given values should be mutable in order for TransformerServable classes to update
     * DataFrame with the serving results.
     */
    public Row(List<Object> values) {
        this.values = values;
    }

    /** Returns the value at the given index. */
    public Object get(int index) {
        return values.get(index);
    }

    /** Returns the value at the given index as the given type. */
    @SuppressWarnings("unchecked")
    public <T> T getAs(int index) {
        return (T) values.get(index);
    }

    /** Adds the value to the end of this row and returns this row. */
    public Row add(Object value) {
        values.add(value);
        return this;
    }

    /** Returns the number of values in this row. */
    public int size() {
        return values.size();
    }
}
