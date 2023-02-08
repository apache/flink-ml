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

package org.apache.flink.ml.common.util;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Tests {@link QuantileSummary}. */
public class QuantileSummaryTest {

    private List<double[]> datasets;

    @Before
    public void prepare() {
        double[] increasing = IntStream.range(0, 100).mapToDouble(x -> x).toArray();
        double[] decreasing = IntStream.range(0, 100).mapToDouble(x -> 99 - x).toArray();
        double[] negatives = IntStream.range(-100, 0).mapToDouble(x -> x).toArray();

        datasets = new ArrayList<>(Arrays.asList(increasing, decreasing, negatives));
    }

    private QuantileSummary buildSummary(double[] data, double epsilon) {
        QuantileSummary summary = new QuantileSummary(epsilon);
        for (double datum : data) {
            summary = summary.insert(datum);
        }
        return summary.compress();
    }

    private void checkQuantiles(double[] data, double[] percentiles, QuantileSummary summary) {
        if (data.length == 0) {
            assertNull(summary.query(percentiles));
        } else {
            double[] quantiles = summary.query(percentiles);
            IntStream.range(0, percentiles.length)
                    .forEach(
                            i ->
                                    validateApproximation(
                                            quantiles[i], data, percentiles[i], summary));
        }
    }

    private void validateApproximation(
            double approx, double[] data, double percentile, QuantileSummary summary) {
        double rank =
                Math.ceil(
                        (Arrays.stream(data).filter(x -> x <= approx).count()
                                        + Arrays.stream(data).filter(x -> x < approx).count())
                                / 2.0);
        double lower = Math.floor((percentile - summary.getRelativeError()) * data.length);
        double upper =
                summary.getRelativeError() == 0
                        ? Math.ceil((percentile + summary.getRelativeError()) * data.length) + 1
                        : Math.ceil((percentile + summary.getRelativeError()) * data.length);
        String errMessage =
                String.format(
                        "Rank not in [%s, %s], percentile: %s, approx returned: %s",
                        lower, upper, percentile, approx);
        assertTrue(errMessage, rank >= lower && rank <= upper);
    }

    private void checkMergedQuantiles(
            double[] data1,
            double epsilon1,
            double[] data2,
            double epsilon2,
            double[] percentiles) {
        QuantileSummary summary1 = buildSummary(data1, epsilon1);
        QuantileSummary summary2 = buildSummary(data2, epsilon2);
        QuantileSummary newSummary = summary2.merge(summary1);

        double[] quantiles = newSummary.query(percentiles);
        IntStream.range(0, percentiles.length)
                .forEach(
                        i ->
                                validateApproximation(
                                        quantiles[i],
                                        ArrayUtils.addAll(data1, data2),
                                        percentiles[i],
                                        newSummary));
    }

    @Test
    public void testQuantiles() {
        for (double[] data : datasets) {
            QuantileSummary summary = buildSummary(data, 0.001);
            double[] percentiles = {0, 0.01, 0.1, 0.25, 0.75, 0.5, 0.9, 0.99, 1};
            checkQuantiles(data, percentiles, summary);
        }
    }

    @Test
    public void testNoRelativeError() {
        for (double[] data : datasets) {
            QuantileSummary summary = buildSummary(data, 0.0);
            double[] percentiles = {0, 0.01, 0.1, 0.25, 0.75, 0.5, 0.9, 0.99, 1};
            checkQuantiles(data, percentiles, summary);
        }
    }

    @Test
    public void testOnEmptyDataset() {
        double[] data = new double[0];
        QuantileSummary summary = buildSummary(data, 0.001);
        double[] percentiles = {0, 0.01, 0.1, 0.25, 0.75, 0.5, 0.9, 0.99, 1};
        try {
            checkQuantiles(data, percentiles, summary);
            fail();
        } catch (Throwable e) {
            assertEquals("Cannot query percentiles without any records inserted.", e.getMessage());
        }
    }

    @Test
    public void testMerge() {
        double[] data1 = IntStream.range(0, 100).mapToDouble(x -> x).toArray();
        double[] data2 = IntStream.range(100, 200).mapToDouble(x -> x).toArray();
        double[] data3 = IntStream.range(0, 1000).mapToDouble(x -> x).toArray();
        double[] data4 = IntStream.range(-50, 50).mapToDouble(x -> x).toArray();

        double[] percentiles = {0, 0.01, 0.1, 0.25, 0.75, 0.5, 0.9, 0.99, 1};
        checkMergedQuantiles(data1, 0.001, data2, 0.001, percentiles);
        checkMergedQuantiles(data1, 0.0001, data2, 0.0001, percentiles);
        checkMergedQuantiles(data1, 0.001, data3, 0.001, percentiles);
        checkMergedQuantiles(data1, 0.001, data4, 0.001, percentiles);
    }

    @Test
    public void testQuerySinglePercentile() {
        QuantileSummary summary = buildSummary(datasets.get(0), 0.001);
        double approx = summary.query(0.25);
        validateApproximation(approx, datasets.get(0), 0.25, summary);
    }

    @Test
    public void testCompressMultiTimes() {
        QuantileSummary summary = buildSummary(datasets.get(0), 0.001);
        QuantileSummary newSummary = summary.compress();
        assertEquals(summary, newSummary);
    }

    @Test
    public void testIsEmpty() {
        QuantileSummary summary = new QuantileSummary(0.01);
        assertTrue(summary.isEmpty());

        summary = summary.insert(1);
        assertFalse(summary.isEmpty());

        summary = summary.compress();
        assertFalse(summary.isEmpty());

        summary = summary.merge(new QuantileSummary(0.01));
        assertFalse(summary.isEmpty());
    }
}
