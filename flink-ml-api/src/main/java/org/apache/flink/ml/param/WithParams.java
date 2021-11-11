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
import org.apache.flink.ml.util.ParamUtils;

import java.util.Map;
import java.util.Optional;

/**
 * Interface for classes that take parameters. It provides APIs to set and get parameters.
 *
 * @param <T> The class type of WithParams implementation itself.
 */
@PublicEvolving
public interface WithParams<T> {

    /**
     * Gets the parameter by its name.
     *
     * @param name The parameter name.
     * @param <V> The class type of the parameter value.
     * @return The parameter.
     */
    default <V> Param<V> getParam(String name) {
        Optional<Param<?>> result =
                getParamMap().keySet().stream().filter(param -> param.name.equals(name)).findAny();
        return (Param<V>) result.orElse(null);
    }

    /**
     * Sets the value of the parameter.
     *
     * @param param The parameter.
     * @param value The parameter value.
     * @return The WithParams instance itself.
     */
    @SuppressWarnings("unchecked")
    default <V> T set(Param<V> param, V value) {
        if (value != null && !param.clazz.isAssignableFrom(value.getClass())) {
            throw new ClassCastException(
                    "Parameter "
                            + param.name
                            + " is given a value with incompatible class "
                            + value.getClass().getName());
        }

        if (!param.validator.validate(value)) {
            if (value == null) {
                throw new IllegalArgumentException(
                        "Parameter " + param.name + "'s value should not be null");
            } else {
                throw new IllegalArgumentException(
                        "Parameter "
                                + param.name
                                + " is given an invalid value "
                                + value.toString());
            }
        }
        getParamMap().put(param, value);
        return (T) this;
    }

    /**
     * Gets the value of the parameter.
     *
     * @param param The parameter.
     * @param <V> The class type of the parameter value.
     * @return The parameter value.
     */
    @SuppressWarnings("unchecked")
    default <V> V get(Param<V> param) {
        Map<Param<?>, Object> paramMap = getParamMap();
        V value = (V) paramMap.get(param);

        if (value == null && !param.validator.validate(value)) {
            throw new IllegalArgumentException(
                    "Parameter " + param.name + "'s value should not be null");
        }

        return value;
    }

    /**
     * Returns a map which should contain value for every parameter that meets one of the following
     * conditions.
     *
     * <p>1) set(...) has been called to set value for this parameter.
     *
     * <p>2) The parameter is a public final field of this WithParams instance. This includes fields
     * inherited from its interfaces and super-classes.
     *
     * <p>The subclass which implements this interface could meet this requirement by returning a
     * member field of the given map type, after having initialized this member field using the
     * {@link ParamUtils#initializeMapWithDefaultValues(Map, WithParams)} method.
     *
     * @return A map which maps parameter definition to parameter value.
     */
    Map<Param<?>, Object> getParamMap();
}
