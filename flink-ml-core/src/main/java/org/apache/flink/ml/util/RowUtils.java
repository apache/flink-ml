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

import org.apache.flink.types.Row;
import org.apache.flink.types.RowKind;

/** Utilities to deal with {@link Row} instances. */
public class RowUtils {

    /**
     * Creates a new row with fields that are copied from the existing one, and reserves some empty
     * fields for further writing. The {@link RowKind} of the first row determines the {@link
     * RowKind} of the result.
     *
     * @param existing The existing row to be cloned.
     * @param reservedFields Num of fields to be reserved.
     */
    public static Row cloneWithReservedFields(Row existing, int reservedFields) {
        Row result = new Row(existing.getKind(), existing.getArity() + reservedFields);
        for (int i = 0; i < existing.getArity(); i++) {
            result.setField(i, existing.getField(i));
        }
        return result;
    }

    /**
     * Creates a new row with fields that are copied from the existing one, and appends a new filed
     * with the specific value.
     *
     * @param existing The existing row to be cloned.
     * @param value The value to be appended.
     */
    public static Row append(Row existing, Object value) {
        Row result = cloneWithReservedFields(existing, 1);
        result.setField(existing.getArity(), value);
        return result;
    }
}
