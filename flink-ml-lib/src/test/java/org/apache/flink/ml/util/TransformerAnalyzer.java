/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.util;

import org.apache.flink.client.ClientUtils;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.util.JarUtils;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Optional;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/** Utils to analyze the stage defined in the jar. */
public class TransformerAnalyzer {

    private static final String CLASS_EXTENSION = ".class";

    public static List<String> analyzeLibJars(String path) throws IOException {
        URL jarURL = JarUtils.getJarFiles(new String[] {path}).get(0);
        JarFile jar = new JarFile(new File(jarURL.getPath()));

        Enumeration<JarEntry> entries = jar.entries();

        URLClassLoader loader =
                ClientUtils.buildUserCodeClassLoader(
                        Collections.singletonList(jarURL),
                        Collections.emptyList(),
                        TransformerAnalyzer.class.getClassLoader(),
                        new Configuration());

        List<String> transformers = new ArrayList<>();
        while (entries.hasMoreElements()) {
            JarEntry jarEntry = entries.nextElement();

            if (isNotClass(jarEntry)) {
                continue;
            }

            Optional<Class<?>> optionalClass = loadClass(jarEntry, loader);

            if (!optionalClass.map(TransformerAnalyzer::isInstantiableTransformer).orElse(false)) {
                continue;
            }

            Class<?> clazz = optionalClass.get();
            transformers.add(clazz.getName());
        }
        return transformers;
    }

    private static boolean isNotClass(JarEntry jarEntry) {
        return jarEntry.isDirectory() || !jarEntry.getName().endsWith(CLASS_EXTENSION);
    }

    private static Optional<Class<?>> loadClass(JarEntry jarEntry, ClassLoader classLoader) {
        String className =
                jarEntry.getName()
                        .substring(0, jarEntry.getName().length() - CLASS_EXTENSION.length());
        className = className.replace('/', '.');
        try {
            return Optional.of(classLoader.loadClass(className));
        } catch (Throwable t) {
            System.err.println(
                    String.format(
                            "Failed to load class %s while analyzing flink-ml-lib JAR because of %s.",
                            className, t));
            return Optional.empty();
        }
    }

    private static boolean isInstantiableTransformer(Class<?> clazz) {
        if (!Transformer.class.isAssignableFrom(clazz)) {
            return false;
        }
        // check that class is non-abstract and public
        int mods = clazz.getModifiers();
        if (Modifier.isAbstract(mods) || !Modifier.isPublic(mods)) {
            return false;
        }

        // check that class has accessible default constructor
        try {
            Constructor<?> defaultConstructor = clazz.getConstructor();
            int constructorMods = defaultConstructor.getModifiers();
            return Modifier.isPublic(constructorMods);
        } catch (Throwable t) {
            System.err.println(
                    String.format(
                            "Failed to load class {} while analyzing flink-ml-lib JAR because of %s.",
                            clazz.getCanonicalName()));
            return false;
        }
    }
}
