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

package org.apache.flink.ml.api.core;

import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.DoubleArrayParam;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.FloatArrayParam;
import org.apache.flink.ml.param.FloatParam;
import org.apache.flink.ml.param.IntArrayParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.LongArrayParam;
import org.apache.flink.ml.param.LongParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidator;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringArrayParam;
import org.apache.flink.ml.param.StringParam;
import org.apache.flink.ml.param.WithParams;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;

import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/** Tests the behavior of Stage and WithParams. */
public class StageTest {

    // A WithParams subclass which has one parameter for each pre-defined parameter type.
    private interface MyParams<T> extends WithParams<T> {
        Param<Boolean> BOOLEAN_PARAM = new BooleanParam("booleanParam", "Description", false);

        Param<Integer> INT_PARAM =
                new IntParam("intParam", "Description", 1, ParamValidators.lt(100));

        Param<Long> LONG_PARAM =
                new LongParam("longParam", "Description", 2L, ParamValidators.lt(100));

        Param<Float> FLOAT_PARAM =
                new FloatParam("floatParam", "Description", 3.0f, ParamValidators.lt(100));

        Param<Double> DOUBLE_PARAM =
                new DoubleParam("doubleParam", "Description", 4.0, ParamValidators.lt(100));

        Param<String> STRING_PARAM = new StringParam("stringParam", "Description", "5");

        Param<Integer[]> INT_ARRAY_PARAM =
                new IntArrayParam("intArrayParam", "Description", new Integer[] {6, 7});

        Param<Long[]> LONG_ARRAY_PARAM =
                new LongArrayParam(
                        "longArrayParam",
                        "Description",
                        new Long[] {8L, 9L},
                        ParamValidators.alwaysTrue());

        Param<Float[]> FLOAT_ARRAY_PARAM =
                new FloatArrayParam("floatArrayParam", "Description", new Float[] {10.0f, 11.0f});

        Param<Double[]> DOUBLE_ARRAY_PARAM =
                new DoubleArrayParam(
                        "doubleArrayParam",
                        "Description",
                        new Double[] {12.0, 13.0},
                        ParamValidators.alwaysTrue());

        Param<String[]> STRING_ARRAY_PARAM =
                new StringArrayParam("stringArrayParam", "Description", new String[] {"14", "15"});
    }

    /**
     * A Stage subclass which inherits all parameters from MyParams and defines an extra parameter.
     */
    public static class MyStage implements Stage<MyStage>, MyParams<MyStage> {
        private final Map<Param<?>, Object> paramMap = new HashMap<>();

        public final Param<Integer> extraIntParam =
                new IntParam("extraIntParam", "Description", 20, ParamValidators.alwaysTrue());

        public final Param<Integer> paramWithNullDefault =
                new IntParam(
                        "paramWithNullDefault",
                        "Must be explicitly set with a non-null value",
                        null,
                        ParamValidators.notNull());

        public MyStage() {
            ParamUtils.initializeMapWithDefaultValues(paramMap, this);
        }

        @Override
        public Map<Param<?>, Object> getParamMap() {
            return paramMap;
        }

        @Override
        public void save(String path) throws IOException {
            ReadWriteUtils.saveMetadata(this, path);
        }

        public static MyStage load(String path) throws IOException {
            return ReadWriteUtils.loadStageParam(path);
        }
    }

    /** A Stage subclass without the static load() method. */
    public static class MyStageWithoutLoad implements Stage<MyStage>, MyParams<MyStage> {
        private final Map<Param<?>, Object> paramMap = new HashMap<>();

        public MyStageWithoutLoad() {
            ParamUtils.initializeMapWithDefaultValues(paramMap, this);
        }

        @Override
        public void save(String path) throws IOException {
            ReadWriteUtils.saveMetadata(this, path);
        }

        @Override
        public Map<Param<?>, Object> getParamMap() {
            return paramMap;
        }
    }

    // Asserts that m1 and m2 are equivalent.
    private static void assertParamMapEquals(Map<Param<?>, Object> m1, Map<Param<?>, Object> m2) {
        Assert.assertTrue(m1 != null && m2 != null);
        Assert.assertEquals(m1.size(), m2.size());

        for (Map.Entry<Param<?>, Object> entry : m1.entrySet()) {
            Assert.assertTrue(m2.containsKey(entry.getKey()));
            Object v1 = entry.getValue();
            Object v2 = m2.get(entry.getKey());
            if (v1 == null || v2 == null) {
                Assert.assertTrue(v1 == null && v2 == null);
            } else if (v1.getClass().isArray() && v2.getClass().isArray()) {
                Assert.assertArrayEquals((Object[]) v1, (Object[]) v2);
            } else {
                Assert.assertEquals(v1, v2);
            }
        }
    }

    // Saves and loads the given stage. And verifies that the loaded stage has same parameter values
    // as the original stage.
    private static Stage<?> validateStageSaveLoad(
            Stage<?> stage, Map<String, Object> paramOverrides) throws IOException {
        for (Map.Entry<String, Object> entry : paramOverrides.entrySet()) {
            Param<?> param = stage.getParam(entry.getKey());
            ReadWriteUtils.setStageParam(stage, param, entry.getValue());
        }

        String tempDir = Files.createTempDirectory("").toString();
        String path = Paths.get(tempDir, "test").toString();
        stage.save(path);
        try {
            stage.save(path);
            Assert.fail("Expected IOException");
        } catch (IOException e) {
            // This is expected.
        }

        Stage<?> loadedStage = ReadWriteUtils.loadStage(path);
        for (Map.Entry<String, Object> entry : paramOverrides.entrySet()) {
            Param<?> param = loadedStage.getParam(entry.getKey());
            Assert.assertEquals(entry.getValue(), loadedStage.get(param));
        }
        assertParamMapEquals(stage.getParamMap(), loadedStage.getParamMap());
        return loadedStage;
    }

    @Test
    public void testParamSetValueWithName() {
        MyStage stage = new MyStage();

        Param<Integer> paramA = MyParams.INT_PARAM;
        stage.set(paramA, 2);
        Assert.assertEquals(2, (int) stage.get(paramA));

        Param<Integer> paramB = stage.getParam("intParam");
        stage.set(paramB, 3);
        Assert.assertEquals(3, (int) stage.get(paramB));

        Param<Integer> paramC = stage.getParam("extraIntParam");
        stage.set(paramC, 50);
        Assert.assertEquals(50, (int) stage.get(paramC));
    }

    @Test
    public void testParamWithNullDefault() {
        MyStage stage = new MyStage();
        try {
            stage.get(stage.paramWithNullDefault);
            Assert.fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            Assert.assertTrue(e.getMessage().contains("should not be null"));
        }

        stage.set(stage.paramWithNullDefault, 3);
        Assert.assertEquals(3, (int) stage.get(stage.paramWithNullDefault));
    }

    private static <T> void assertInvalidValue(Stage<?> stage, Param<T> param, T value) {
        try {
            stage.set(param, value);
            Assert.fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            Assert.assertTrue(e.getMessage().contains("invalid value"));
        }
    }

    private static <T> void assertInvalidClass(Stage<?> stage, Param<T> param, Object value) {
        try {
            stage.set(param, (T) value);
            Assert.fail("Expected ClassCastException");
        } catch (ClassCastException e) {
            Assert.assertTrue(e.getMessage().contains("incompatible class"));
        }
    }

    @Test
    public void testParamSetInvalidValue() {
        MyStage stage = new MyStage();
        assertInvalidValue(stage, MyParams.INT_PARAM, 100);
        assertInvalidValue(stage, MyParams.LONG_PARAM, 100L);
        assertInvalidValue(stage, MyParams.FLOAT_PARAM, 100.0f);
        assertInvalidValue(stage, MyParams.DOUBLE_PARAM, 100.0);
        assertInvalidClass(stage, MyParams.INT_PARAM, "100");
        assertInvalidClass(stage, MyParams.STRING_PARAM, 100);

        Param<Integer> param = stage.getParam("stringParam");
        assertInvalidClass(stage, param, 50);
    }

    @Test
    public void testParamSetValidValue() {
        MyStage stage = new MyStage();

        stage.set(MyParams.BOOLEAN_PARAM, true);
        Assert.assertEquals(true, stage.get(MyParams.BOOLEAN_PARAM));

        stage.set(MyParams.INT_PARAM, 50);
        Assert.assertEquals(50, (int) stage.get(MyParams.INT_PARAM));

        stage.set(MyParams.LONG_PARAM, 50L);
        Assert.assertEquals(50L, (long) stage.get(MyParams.LONG_PARAM));

        stage.set(MyParams.FLOAT_PARAM, 50f);
        Assert.assertEquals(50f, (float) stage.get(MyParams.FLOAT_PARAM), 0.0001);

        stage.set(MyParams.DOUBLE_PARAM, 50.0);
        Assert.assertEquals(50, (double) stage.get(MyParams.DOUBLE_PARAM), 0.0001);

        stage.set(MyParams.STRING_PARAM, "50");
        Assert.assertEquals("50", stage.get(MyParams.STRING_PARAM));

        stage.set(MyParams.INT_ARRAY_PARAM, new Integer[] {50, 51});
        Assert.assertArrayEquals(new Integer[] {50, 51}, stage.get(MyParams.INT_ARRAY_PARAM));

        stage.set(MyParams.LONG_ARRAY_PARAM, new Long[] {50L, 51L});
        Assert.assertArrayEquals(new Long[] {50L, 51L}, stage.get(MyParams.LONG_ARRAY_PARAM));

        stage.set(MyParams.FLOAT_ARRAY_PARAM, new Float[] {50.0f, 51.0f});
        Assert.assertArrayEquals(new Float[] {50.0f, 51.0f}, stage.get(MyParams.FLOAT_ARRAY_PARAM));

        stage.set(MyParams.DOUBLE_ARRAY_PARAM, new Double[] {50.0, 51.0});
        Assert.assertArrayEquals(new Double[] {50.0, 51.0}, stage.get(MyParams.DOUBLE_ARRAY_PARAM));

        stage.set(MyParams.STRING_ARRAY_PARAM, new String[] {"50", "51"});
        Assert.assertArrayEquals(new String[] {"50", "51"}, stage.get(MyParams.STRING_ARRAY_PARAM));
    }

    @Test
    public void testStageSaveLoad() throws IOException {
        MyStage stage = new MyStage();
        stage.set(stage.paramWithNullDefault, 1);
        Stage<?> loadedStage = validateStageSaveLoad(stage, Collections.emptyMap());
        Assert.assertEquals(1, (int) loadedStage.get(MyParams.INT_PARAM));
    }

    @Test
    public void testStageSaveLoadWithParamOverrides() throws IOException {
        MyStage stage = new MyStage();
        stage.set(stage.paramWithNullDefault, 1);
        Stage<?> loadedStage =
                validateStageSaveLoad(stage, Collections.singletonMap("intParam", 10));
        Assert.assertEquals(10, (int) loadedStage.get(MyParams.INT_PARAM));
    }

    @Test
    public void testStageLoadWithoutLoadMethod() throws IOException {
        MyStageWithoutLoad stage = new MyStageWithoutLoad();
        try {
            validateStageSaveLoad(stage, Collections.emptyMap());
            Assert.fail("Expected RuntimeException");
        } catch (RuntimeException e) {
            Assert.assertTrue(e.getMessage().contains("not implemented"));
        }
    }

    @Test
    public void testValidators() {
        ParamValidator<Integer> gt = ParamValidators.gt(10);
        Assert.assertFalse(gt.validate(null));
        Assert.assertFalse(gt.validate(5));
        Assert.assertFalse(gt.validate(10));
        Assert.assertTrue(gt.validate(15));

        ParamValidator<Integer> gtEq = ParamValidators.gtEq(10);
        Assert.assertFalse(gtEq.validate(null));
        Assert.assertFalse(gtEq.validate(5));
        Assert.assertTrue(gtEq.validate(10));
        Assert.assertTrue(gtEq.validate(15));

        ParamValidator<Integer> lt = ParamValidators.lt(10);
        Assert.assertFalse(lt.validate(null));
        Assert.assertTrue(lt.validate(5));
        Assert.assertFalse(lt.validate(10));
        Assert.assertFalse(lt.validate(15));

        ParamValidator<Integer> ltEq = ParamValidators.ltEq(10);
        Assert.assertFalse(ltEq.validate(null));
        Assert.assertTrue(ltEq.validate(5));
        Assert.assertTrue(ltEq.validate(10));
        Assert.assertFalse(ltEq.validate(15));

        ParamValidator<Integer> inRangeInclusive = ParamValidators.inRange(5, 15);
        Assert.assertFalse(inRangeInclusive.validate(null));
        Assert.assertFalse(inRangeInclusive.validate(0));
        Assert.assertTrue(inRangeInclusive.validate(5));
        Assert.assertTrue(inRangeInclusive.validate(10));
        Assert.assertTrue(inRangeInclusive.validate(15));
        Assert.assertFalse(inRangeInclusive.validate(20));

        ParamValidator<Integer> inRangeExclusive = ParamValidators.inRange(5, 15, false, false);
        Assert.assertFalse(inRangeExclusive.validate(null));
        Assert.assertFalse(inRangeExclusive.validate(0));
        Assert.assertFalse(inRangeExclusive.validate(5));
        Assert.assertTrue(inRangeExclusive.validate(10));
        Assert.assertFalse(inRangeExclusive.validate(15));
        Assert.assertFalse(inRangeExclusive.validate(20));

        ParamValidator<Integer> inArray = ParamValidators.inArray(1, 2, 3);
        Assert.assertFalse(inArray.validate(null));
        Assert.assertTrue(inArray.validate(1));
        Assert.assertFalse(inArray.validate(0));

        ParamValidator<Integer> notNull = ParamValidators.notNull();
        Assert.assertTrue(notNull.validate(5));
        Assert.assertFalse(notNull.validate(null));
    }
}
