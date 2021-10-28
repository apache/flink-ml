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

import org.apache.flink.annotation.PublicEvolving;

import java.io.Serializable;

/**
 * An interface to validate that a parameter value is valid.
 *
 * @param <T> The class type of the parameter value.
 */
@PublicEvolving
public interface ParamValidator<T> extends Serializable {

    /**
     * Validate whether the parameter value is valid.
     *
     * @param value The parameter value.
     * @return A boolean indicating whether the parameter value is valid.
     */
    boolean validate(T value);
}
