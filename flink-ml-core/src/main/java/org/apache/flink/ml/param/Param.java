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

import java.io.IOException;
import java.io.Serializable;

/**
 * Definition of a parameter, including name, class, description, default value and the validator.
 *
 * @param <T> The class type of the parameter value.
 */
@PublicEvolving
public class Param<T> implements Serializable {
    private static final long serialVersionUID = 4396556083935765299L;

    public final String name;
    public final Class<T> clazz;
    public final String description;
    public final T defaultValue;
    public final ParamValidator<T> validator;

    public Param(
            String name,
            Class<T> clazz,
            String description,
            T defaultValue,
            ParamValidator<T> validator) {
        this.name = name;
        this.clazz = clazz;
        this.description = description;
        this.defaultValue = defaultValue;
        this.validator = validator;

        if (defaultValue != null && !validator.validate(defaultValue)) {
            throw new IllegalArgumentException(
                    "Parameter " + name + " is given an invalid value " + defaultValue);
        }
    }

    /**
     * Encodes the given object into a json-formatted string.
     *
     * @param value An object of class type T.
     * @return A json-formatted string.
     */
    public String jsonEncode(T value) throws IOException {
        return ReadWriteUtils.OBJECT_MAPPER.writeValueAsString(value);
    }

    /**
     * Decodes the given string into an object of class type T.
     *
     * @param json A json-formatted string.
     * @return An object of class type T.
     */
    @SuppressWarnings("unchecked")
    public T jsonDecode(String json) throws IOException {
        return ReadWriteUtils.OBJECT_MAPPER.readValue(json, clazz);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Param)) {
            return false;
        }
        return ((Param<?>) obj).name.equals(name);
    }

    @Override
    public int hashCode() {
        return name.hashCode();
    }

    @Override
    public String toString() {
        return name;
    }
}
