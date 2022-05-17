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

package org.apache.flink.ml.param;

import org.apache.commons.lang3.ArrayUtils;

/** Factory methods for common validation functions on numerical values. */
public class ParamValidators {

    // Always return true.
    public static <T> ParamValidator<T> alwaysTrue() {
        return (value) -> true;
    }

    // Checks if the parameter value is greater than lowerBound.
    public static <T> ParamValidator<T> gt(double lowerBound) {
        return (value) -> value != null && ((Number) value).doubleValue() > lowerBound;
    }

    // Checks if the parameter value is greater than or equal to lowerBound.
    public static <T> ParamValidator<T> gtEq(double lowerBound) {
        return (value) -> value != null && ((Number) value).doubleValue() >= lowerBound;
    }

    // Checks if the parameter value is less than upperBound.
    public static <T> ParamValidator<T> lt(double upperBound) {
        return (value) -> value != null && ((Number) value).doubleValue() < upperBound;
    }

    // Checks if the parameter value is less than or equal to upperBound.
    public static <T> ParamValidator<T> ltEq(double upperBound) {
        return (value) -> value != null && ((Number) value).doubleValue() <= upperBound;
    }

    /**
     * Check if the parameter value is in the range from lowerBound to upperBound.
     *
     * @param lowerInclusive if true, range includes value = lowerBound
     * @param upperInclusive if true, range includes value = upperBound
     */
    public static <T> ParamValidator<T> inRange(
            double lowerBound, double upperBound, boolean lowerInclusive, boolean upperInclusive) {
        return new ParamValidator<T>() {
            @Override
            public boolean validate(T obj) {
                if (obj == null) {
                    return false;
                }
                double value = ((Number) obj).doubleValue();
                return (value >= lowerBound)
                        && (value <= upperBound)
                        && (lowerInclusive || value != lowerBound)
                        && (upperInclusive || value != upperBound);
            }
        };
    }

    // Checks if the parameter value is in the range [lowerBound, upperBound].
    public static <T> ParamValidator<T> inRange(double lowerBound, double upperBound) {
        return inRange(lowerBound, upperBound, true, true);
    }

    // Checks if the parameter value is in the array of allowed values.
    public static <T> ParamValidator<T> inArray(T... allowed) {
        return new ParamValidator<T>() {
            @Override
            public boolean validate(T value) {
                return value != null && ArrayUtils.contains(allowed, value);
            }
        };
    }

    // Checks if the parameter value is not null.
    public static <T> ParamValidator<T> notNull() {
        return new ParamValidator<T>() {
            @Override
            public boolean validate(T value) {
                return value != null;
            }
        };
    }

    // Checks if the parameter value array is not empty array.
    public static <T> ParamValidator<T[]> nonEmptyArray() {
        return value -> value != null && value.length > 0;
    }

    // Checks if every element in the array-typed parameter value is in the array of allowed values.
    @SafeVarargs
    public static <T> ParamValidator<T[]> isSubSet(T... allowed) {
        return new ParamValidator<T[]>() {
            @Override
            public boolean validate(T[] value) {
                if (value == null) {
                    return false;
                }
                for (T t : value) {
                    if (!ArrayUtils.contains(allowed, t)) {
                        return false;
                    }
                }
                return true;
            }
        };
    }
}
