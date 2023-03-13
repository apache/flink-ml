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

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.WithParams;
import org.apache.flink.util.InstantiationUtil;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Utility methods for reading and writing stages. */
public class ParamUtils {
    /**
     * Updates the paramMap with default values of all public final Param-typed fields of the given
     * instance. A parameter's value will not be updated if this parameter is already found in the
     * map.
     *
     * <p>Note: This method should be called after all public final Param-typed fields of the given
     * instance have been defined. A good choice is to call this method in the constructor of the
     * given instance.
     */
    public static void initializeMapWithDefaultValues(
            Map<Param<?>, Object> paramMap, WithParams<?> instance) {
        List<Param<?>> defaultParams = getPublicFinalParamFields(instance);
        for (Param<?> param : defaultParams) {
            if (!paramMap.containsKey(param)) {
                paramMap.put(param, param.defaultValue);
            }
        }
    }

    /**
     * Finds all public final fields of the Param class type of the given object, including those
     * fields inherited from its interfaces and super-classes, and returns those Param instances as
     * a list.
     *
     * @param object the object whose public final Param-typed fields will be returned.
     * @return a list of Param instances.
     */
    public static List<Param<?>> getPublicFinalParamFields(Object object) {
        return getPublicFinalParamFields(object, object.getClass());
    }

    // A helper method that finds all public final fields of the Param class type of the given
    // object and returns those Param instances as a list. The clazz specifies the object class.
    private static List<Param<?>> getPublicFinalParamFields(Object object, Class<?> clazz) {
        List<Param<?>> result = new ArrayList<>();
        for (Field field : clazz.getDeclaredFields()) {
            field.setAccessible(true);
            if (Param.class.isAssignableFrom(field.getType())
                    && Modifier.isPublic(field.getModifiers())
                    && Modifier.isFinal(field.getModifiers())) {
                try {
                    result.add((Param<?>) field.get(object));
                } catch (IllegalAccessException e) {
                    throw new RuntimeException(
                            "Failed to extract param from field " + field.getName(), e);
                }
            }
        }

        if (clazz.getSuperclass() != null) {
            result.addAll(getPublicFinalParamFields(object, clazz.getSuperclass()));
        }
        for (Class<?> cls : clazz.getInterfaces()) {
            result.addAll(getPublicFinalParamFields(object, cls));
        }
        return result;
    }

    // A helper method that sets WithParams object's parameter value. We can not call
    // WithParams.set(param, value)
    // directly because WithParams::set(...) needs the actual type of the value.
    @SuppressWarnings("unchecked")
    public static <T> void setParam(WithParams<?> instance, Param<T> param, Object value) {
        instance.set(param, (T) value);
    }

    // A helper method that updates WithParams instance's param map using values from the
    // paramOverrides. This method only updates values for parameters already defined in the
    // instance's param map.
    public static void updateExistingParams(
            WithParams<?> instance, Map<Param<?>, Object> paramOverrides) {
        Set<Param<?>> existingParams = instance.getParamMap().keySet();
        for (Map.Entry<Param<?>, Object> entry : paramOverrides.entrySet()) {
            if (existingParams.contains(entry.getKey())) {
                ParamUtils.setParam(instance, entry.getKey(), entry.getValue());
            }
        }
    }

    /**
     * Instantiates a WithParams subclass from the provided json map.
     *
     * @param jsonMap a map containing className and paramMap.
     * @return the instantiated WithParams subclass instance.
     */
    @SuppressWarnings("unchecked")
    public static <T extends WithParams<T>> T instantiateWithParams(Map<String, ?> jsonMap)
            throws ClassNotFoundException, IOException {
        String className = (String) jsonMap.get("className");
        Class<T> clazz = (Class<T>) Class.forName(className);
        T instance = InstantiationUtil.instantiate(clazz);

        Map<String, Param<?>> nameToParam = new HashMap<>();
        for (Param<?> param : ParamUtils.getPublicFinalParamFields(instance)) {
            nameToParam.put(param.name, param);
        }

        if (jsonMap.containsKey("paramMap")) {
            Map<String, Object> paramMap = (Map<String, Object>) jsonMap.get("paramMap");
            for (Map.Entry<String, Object> entry : paramMap.entrySet()) {
                Param<?> param = nameToParam.get(entry.getKey());
                ParamUtils.setParam(instance, param, param.jsonDecode(entry.getValue()));
            }
        }

        return instance;
    }
}
