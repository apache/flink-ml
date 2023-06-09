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

package org.apache.flink.iteration.utils;

import org.apache.flink.util.Preconditions;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Utility class to provide some reflection tools. */
public class ReflectionUtils {

    public static Field getClassField(Class<?> declaredClass, String fieldName) {
        try {
            Field field = declaredClass.getDeclaredField(fieldName);
            field.setAccessible(true);
            return field;
        } catch (Exception e) {
            throw new RuntimeException(
                    "Failed to get field" + fieldName + " from " + declaredClass, e);
        }
    }

    @SuppressWarnings("unchecked")
    public static <T> T getFieldValue(Object targetObject, Field field) {
        try {
            return (T) field.get(targetObject);
        } catch (Exception e) {
            throw new RuntimeException(
                    "Failed to get field " + field.getName() + " from " + targetObject, e);
        }
    }

    public static <T> T getFieldValue(
            Object targetObject, Class<?> declaredClass, String fieldName) {
        Field field = getClassField(declaredClass, fieldName);
        return getFieldValue(targetObject, field);
    }

    public static <T> T callMethod(Object targetObject, Class<?> declaredClass, String methodName) {
        return callMethod(
                targetObject,
                declaredClass,
                methodName,
                Collections.emptyList(),
                Collections.emptyList());
    }

    @SuppressWarnings("unchecked")
    public static <T> T callMethod(
            Object targetObject,
            Class<?> declaredClass,
            String methodName,
            List<Class<?>> parameterClass,
            List<Object> parameters) {

        Method method;
        try {
            method =
                    declaredClass.getDeclaredMethod(
                            methodName, parameterClass.toArray(new Class[0]));
            method.setAccessible(true);
        } catch (NoSuchMethodException e1) {
            try {
                method = declaredClass.getMethod(methodName, parameterClass.toArray(new Class[0]));
            } catch (NoSuchMethodException e2) {
                throw new RuntimeException(
                        "Failed to get method" + methodName + " from " + targetObject, e2);
            }
        }

        try {
            return (T) method.invoke(targetObject, parameters.toArray());
        } catch (Exception e) {
            throw new RuntimeException(
                    "Failed to invoke method" + methodName + " from " + targetObject, e);
        }
    }

    /**
     * The utility method to call method with the specific name. The method can be a public method
     * of the class or one inherited from superclasses and superinterfaces.
     *
     * <p>Note that this method is added only for bypassing the existing bug in Py4j. It doesn't
     * validate the classes of parameters so it can only deal with the classes that have only one
     * method with the specific name.
     *
     * <p>TODO: Remove this method after the Py4j bug is fixed.
     */
    @SuppressWarnings("unchecked")
    public static <T> T callMethod(
            Object targetObject, Class<?> declaredClass, String methodName, Object[] parameters) {
        List<Method> methods = new ArrayList<>();
        for (Method m : declaredClass.getMethods()) {
            if (methodName.equals(m.getName())) {
                methods.add(m);
            }
        }
        Preconditions.checkState(
                methods.size() == 1,
                "Only one method with name %s is permitted to be declared in %s",
                methodName,
                declaredClass);
        try {
            return (T) methods.get(0).invoke(targetObject, parameters);
        } catch (Exception e) {
            throw new RuntimeException(
                    "Failed to invoke method" + methodName + " from " + targetObject, e);
        }
    }
}
