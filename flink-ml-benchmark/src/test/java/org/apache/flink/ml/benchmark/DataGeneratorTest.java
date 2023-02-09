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

package org.apache.flink.ml.benchmark;

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.ml.benchmark.datagenerator.common.DenseVectorArrayGenerator;
import org.apache.flink.ml.benchmark.datagenerator.common.DenseVectorGenerator;
import org.apache.flink.ml.benchmark.datagenerator.common.DoubleGenerator;
import org.apache.flink.ml.benchmark.datagenerator.common.LabeledPointWithWeightGenerator;
import org.apache.flink.ml.benchmark.datagenerator.common.RandomStringArrayGenerator;
import org.apache.flink.ml.benchmark.datagenerator.common.RandomStringGenerator;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** Tests data generators. */
public class DataGeneratorTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamTableEnvironment tEnv;

    @Before
    public void before() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);
    }

    @Test
    public void testDenseVectorGenerator() {
        DenseVectorGenerator generator =
                new DenseVectorGenerator()
                        .setColNames(new String[] {"denseVector"})
                        .setNumValues(100)
                        .setVectorDim(10);

        int count = 0;
        for (CloseableIterator<Row> it = generator.getData(tEnv)[0].execute().collect();
                it.hasNext(); ) {
            Row row = it.next();
            assertEquals(1, row.getArity());
            DenseVector vector = (DenseVector) row.getField(generator.getColNames()[0][0]);
            assertNotNull(vector);
            assertEquals(vector.size(), generator.getVectorDim());
            count++;
        }
        assertEquals(generator.getNumValues(), count);
    }

    @Test
    public void testDenseVectorArrayGenerator() {
        DenseVectorArrayGenerator generator =
                new DenseVectorArrayGenerator()
                        .setColNames(new String[] {"denseVectors"})
                        .setNumValues(100)
                        .setVectorDim(10)
                        .setArraySize(20);

        int count = 0;
        for (CloseableIterator<Row> it = generator.getData(tEnv)[0].execute().collect();
                it.hasNext(); ) {
            Row row = it.next();
            assertEquals(1, row.getArity());
            DenseVector[] vectors = (DenseVector[]) row.getField(generator.getColNames()[0][0]);
            assertNotNull(vectors);
            assertEquals(generator.getArraySize(), vectors.length);
            for (DenseVector vector : vectors) {
                assertEquals(vector.size(), generator.getVectorDim());
            }
            count++;
        }
        assertEquals(generator.getNumValues(), count);
    }

    @Test
    public void testLabeledPointWithWeightGenerator() {
        String featuresCol = "features";
        String labelCol = "label";
        String weightCol = "weight";

        LabeledPointWithWeightGenerator generator =
                new LabeledPointWithWeightGenerator()
                        .setFeatureArity(10)
                        .setLabelArity(10)
                        .setColNames(new String[] {featuresCol, labelCol, weightCol})
                        .setNumValues(100);

        int count = 0;
        for (CloseableIterator<Row> it = generator.getData(tEnv)[0].execute().collect();
                it.hasNext(); ) {
            Row row = it.next();
            count++;
            DenseVector features = (DenseVector) row.getField(featuresCol);
            assertNotNull(features);
            for (double value : features.values) {
                assertTrue(value >= 0);
                assertTrue(value <= generator.getFeatureArity() - 1);
            }

            double label = (double) row.getField(labelCol);
            assertTrue(label >= 0);
            assertTrue(label <= generator.getLabelArity() - 1);

            double weight = (double) row.getField(weightCol);
            assertTrue(weight >= 0);
            assertTrue(weight < 1);
        }
        assertEquals(generator.getNumValues(), count);
    }

    @Test
    public void testRandomStringGenerator() {
        String col1 = "col1";
        String col2 = "col2";

        RandomStringGenerator generator =
                new RandomStringGenerator()
                        .setColNames(new String[] {col1, col2})
                        .setSeed(2L)
                        .setNumValues(5)
                        .setNumDistinctValues(2);

        int count = 0;
        for (CloseableIterator<Row> it = generator.getData(tEnv)[0].execute().collect();
                it.hasNext(); ) {
            Row row = it.next();
            count++;
            String value1 = (String) row.getField(col1);
            String value2 = (String) row.getField(col2);
            assertTrue(Integer.parseInt(value1) < generator.getNumDistinctValues());
            assertTrue(Integer.parseInt(value2) < generator.getNumDistinctValues());
        }
        assertEquals(generator.getNumValues(), count);
    }

    @Test
    public void testRandomStringArrayGenerator() {
        String col1 = "col1";
        String col2 = "col2";

        RandomStringArrayGenerator generator =
                new RandomStringArrayGenerator()
                        .setColNames(new String[] {col1, col2})
                        .setSeed(2L)
                        .setNumValues(5)
                        .setNumDistinctValues(2)
                        .setArraySize(3);

        int count = 0;
        for (CloseableIterator<Row> it = generator.getData(tEnv)[0].execute().collect();
                it.hasNext(); ) {
            Row row = it.next();
            count++;
            String[] value1 = (String[]) row.getField(col1);
            String[] value2 = (String[]) row.getField(col2);
            assertEquals(generator.getArraySize(), value1.length);
            assertEquals(generator.getArraySize(), value2.length);
            for (int i = 0; i < generator.getArraySize(); i++) {
                assertTrue(Integer.parseInt(value1[i]) < generator.getNumDistinctValues());
                assertTrue(Integer.parseInt(value2[i]) < generator.getNumDistinctValues());
            }
        }
        assertEquals(generator.getNumValues(), count);
    }

    @Test
    public void testDoubleGenerator() {
        String col1 = "col1";
        String col2 = "col2";

        DoubleGenerator generator =
                new DoubleGenerator()
                        .setColNames(new String[] {"col1", "col2"})
                        .setSeed(2L)
                        .setNumValues(5);

        int count = 0;
        for (CloseableIterator<Row> it = generator.getData(tEnv)[0].execute().collect();
                it.hasNext(); ) {
            Row row = it.next();
            count++;
            double value1 = (Double) row.getField(col1);
            double value2 = (Double) row.getField(col2);
            assertTrue(value1 <= 1 && value1 >= 0);
            assertTrue(value2 <= 1 && value2 >= 0);
        }
        assertEquals(generator.getNumValues(), count);
    }
}
