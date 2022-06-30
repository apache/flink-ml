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

package org.apache.flink.ml.examples;

import org.apache.flink.test.util.AbstractTestBase;

import org.apache.commons.io.output.NullPrintStream;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/** Extracts all example classes in this package and tests their main methods. */
@RunWith(Parameterized.class)
public class ExamplesTest extends AbstractTestBase {
    private final Method mainMethod;

    private PrintStream originalPrintStream;

    @Before
    public void before() {
        originalPrintStream = System.out;
        System.setOut(new NullPrintStream());
    }

    @After
    public void after() {
        System.setOut(originalPrintStream);
    }

    @Parameterized.Parameters(name = "{0}")
    public static Object[][] testData() throws IOException, ClassNotFoundException {
        List<Object[]> testData = new ArrayList<>();

        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        String packageName = ExamplesTest.class.getPackage().getName();
        URL rootURL =
                Objects.requireNonNull(classLoader.getResource(packageName.replace(".", "/")));
        File rootFile = new File(rootURL.getFile().replace("test-classes", "classes"));
        List<Class<?>> classes = listClasses(packageName, rootFile);

        for (Class<?> clazz : classes) {
            testData.add(new Object[] {clazz});
        }

        return testData.toArray(new Object[0][]);
    }

    private static List<Class<?>> listClasses(String packageName, File rootFile)
            throws ClassNotFoundException {
        List<Class<?>> files = new ArrayList<>();
        for (File file : Objects.requireNonNull(rootFile.listFiles())) {
            if (file.isDirectory()) {
                files.addAll(listClasses(packageName + "." + file.getName(), file));
            } else if (file.getName().endsWith(".class")) {
                String fullName = file.getAbsolutePath().replace("/", ".");
                String className =
                        fullName.substring(
                                fullName.indexOf(packageName),
                                fullName.length() - ".class".length());
                Class<?> clazz = Class.forName(className);
                try {
                    clazz.getMethod("main", String[].class);
                } catch (NoSuchMethodException e) {
                    continue;
                }
                files.add(clazz);
            }
        }
        return files;
    }

    public ExamplesTest(Class<?> clazz) throws NoSuchMethodException {
        mainMethod = clazz.getMethod("main", String[].class);
    }

    @Test
    public void test() throws ExecutionException, InterruptedException {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<String> handler =
                executor.submit(
                        new Callable() {
                            @Override
                            public String call() throws Exception {
                                mainMethod.invoke(null, (Object) null);
                                return null;
                            }
                        });

        try {
            handler.get(5, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            handler.cancel(true);
        }
    }
}
