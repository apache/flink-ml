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
import org.apache.flink.ml.util.ReadWriteUtils;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
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
     * Gets the value of the parameter identified by the given name.
     *
     * @param name The parameter name.
     * @param <V> The class type of the parameter.
     * @return A Param instance.
     */
    default <V> Param<V> getParam(String name) {
        Optional<Param<?>> result =
                getParamMap().keySet().stream().filter(param -> param.name.equals(name)).findAny();
        return (Param<V>) result.orElse(null);
    }

    /**
     * Sets the value of the given parameter in the user-defined map.
     *
     * @param param The parameter definition.
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

        if (value != null && !param.validator.validate((V) value)) {
            throw new IllegalArgumentException(
                    "Parameter " + param.name + " is given an invalid value " + value.toString());
        }
        getUserDefinedParamMap().put(param, value);
        return (T) this;
    }

    /**
     * Gets the value of the given parameter. Returns the value from the user-defined map if
     * set(...) has been explicitly called to set value for this parameter. Otherwise, returns the
     * default value from the definition of this parameter.
     *
     * @param param The parameter.
     * @param <V> The class type of the parameter.
     * @return The value of the parameter.
     */
    @SuppressWarnings("unchecked")
    default <V> V get(Param<V> param) {
        Map<Param<?>, Object> paramMap = getUserDefinedParamMap();
        if (paramMap != null && paramMap.containsKey(param)) {
            return (V) paramMap.get(param);
        }

        return param.defaultValue;
    }

    /**
     * Returns an immutable map that contains value for every parameter that meets one of the
     * following conditions:
     *
     * <p>1) set(...) has been called to set value for this parameter.
     *
     * <p>2) The parameter is a field of this WithParams instance. This includes public, protected
     * and private fields. And this also includes fields inherited from its interfaces and
     * super-classes.
     *
     * @return An immutable map which maps parameter definition to parameter value.
     */
    default Map<Param<?>, Object> getParamMap() {
        Map<Param<?>, Object> paramMap = getUserDefinedParamMap();
        if (paramMap == null) {
            paramMap = new HashMap<>();
        }

        List<Param<?>> defaultParams = ReadWriteUtils.getParamFields(this);
        for (Param<?> param : defaultParams) {
            if (!paramMap.containsKey(param)) {
                paramMap.put(param, param.defaultValue);
            }
        }

        return Collections.unmodifiableMap(paramMap);
    }

    /**
     * Returns a mutable map that can be used to set values for parameters. A subclass of this
     * interface should override this method if it wants to support users to set non-default
     * parameter values.
     *
     * @return a mutable map of parameters and value overrides.
     */
    default Map<Param<?>, Object> getUserDefinedParamMap() {
        return null;
    }
}
