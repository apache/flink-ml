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

import org.apache.flink.api.common.typeinfo.TypeInfo;
import org.apache.flink.ml.common.typeinfo.QuantileSummaryTypeInfoFactory;
import org.apache.flink.util.Preconditions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Helper class to compute an approximate quantile summary. This implementation is based on the
 * algorithm proposed in the paper: "Space-efficient Online Computation of Quantile Summaries" by
 * Greenwald, Michael and Khanna, Sanjeev. (https://doi.org/10.1145/375663.375670)
 */
@TypeInfo(QuantileSummaryTypeInfoFactory.class)
public class QuantileSummary implements Serializable {

    /** The default size of head buffer. */
    private static final int DEFAULT_HEAD_SIZE = 50000;

    /** The default compression threshold. */
    private static final int DEFAULT_COMPRESS_THRESHOLD = 10000;

    /** The target relative error. */
    private double relativeError;

    /**
     * The compression threshold. After the internal buffer of statistics crosses this size, it
     * attempts to compress the statistics together.
     */
    private int compressThreshold;

    /** The count of all the elements inserted to be calculated. */
    private long count;

    /** A buffer of quantile statistics. */
    private List<StatsTuple> sampled;

    /** A buffer of the latest samples seen so far. */
    private List<Double> headBuffer = new ArrayList<>();

    /** Whether the quantile summary has been compressed. */
    private boolean compressed;

    /** Empty QuantileSummary Constructor. */
    public QuantileSummary() {}

    /**
     * QuantileSummary Constructor.
     *
     * @param relativeError The target relative error.
     */
    public QuantileSummary(double relativeError) {
        this(relativeError, DEFAULT_COMPRESS_THRESHOLD);
    }

    /**
     * QuantileSummary Constructor.
     *
     * @param relativeError The target relative error.
     * @param compressThreshold the compression threshold. After the internal buffer of statistics
     *     crosses this size, it attempts to compress the statistics together.
     */
    @SuppressWarnings("unchecked")
    public QuantileSummary(double relativeError, int compressThreshold) {
        this(relativeError, compressThreshold, Collections.EMPTY_LIST, 0, false);
    }

    /**
     * QuantileSummary Constructor.
     *
     * @param relativeError The target relative error.
     * @param compressThreshold the compression threshold.
     * @param sampled A buffer of quantile statistics. See the G-K article for more details.
     * @param count The count of all the elements inserted in the sampled buffer.
     * @param compressed Whether the statistics have been compressed.
     */
    public QuantileSummary(
            double relativeError,
            int compressThreshold,
            List<StatsTuple> sampled,
            long count,
            boolean compressed) {
        Preconditions.checkArgument(
                relativeError >= 0 && relativeError <= 1,
                "An appropriate relative error must be in the range [0, 1].");
        Preconditions.checkArgument(
                compressThreshold > 0, "An compress threshold must greater than 0.");
        this.relativeError = relativeError;
        this.compressThreshold = compressThreshold;
        this.sampled = sampled;
        this.count = count;
        this.compressed = compressed;
    }

    /**
     * Insert a new observation into the summary.
     *
     * @param item The new observation to insert into the summary.
     * @return A summary with the given observation inserted into the summary.
     */
    public QuantileSummary insert(double item) {
        headBuffer.add(item);
        compressed = false;
        if (headBuffer.size() >= DEFAULT_HEAD_SIZE) {
            QuantileSummary result = insertHeadBuffer();
            if (result.sampled.size() >= compressThreshold) {
                return result.compress();
            } else {
                return result;
            }
        } else {
            return this;
        }
    }

    /**
     * Returns a new summary that compresses the summary statistics and the head buffer.
     *
     * <p>This implements the COMPRESS function of the GK algorithm.
     *
     * @return The compressed summary.
     */
    public QuantileSummary compress() {
        if (compressed) {
            return this;
        }
        QuantileSummary inserted = insertHeadBuffer();
        Preconditions.checkState(inserted.headBuffer.isEmpty());
        Preconditions.checkState(inserted.count == count + headBuffer.size());

        List<StatsTuple> compressed =
                compressInternal(inserted.sampled, 2 * relativeError * inserted.count);
        return new QuantileSummary(
                relativeError, compressThreshold, compressed, inserted.count, true);
    }

    /**
     * Merges two summaries together.
     *
     * @param other The summary to be merged.
     * @return The merged summary.
     */
    public QuantileSummary merge(QuantileSummary other) {
        Preconditions.checkState(
                headBuffer.isEmpty(), "Current buffer needs to be compressed before merge.");
        Preconditions.checkState(
                other.headBuffer.isEmpty(), "Other buffer needs to be compressed before merge.");

        if (other.count == 0) {
            return shallowCopy();
        } else if (count == 0) {
            return other.shallowCopy();
        } else {
            List<StatsTuple> mergedSampled = new ArrayList<>();
            double mergedRelativeError = Math.max(relativeError, other.relativeError);
            long mergedCount = count + other.count;
            long additionalSelfDelta =
                    Double.valueOf(Math.floor(2 * other.relativeError * other.count)).longValue();
            long additionalOtherDelta =
                    Double.valueOf(Math.floor(2 * relativeError * count)).longValue();

            int selfIdx = 0;
            int otherIdx = 0;
            while (selfIdx < sampled.size() && otherIdx < other.sampled.size()) {
                StatsTuple selfSample = sampled.get(selfIdx);
                StatsTuple otherSample = other.sampled.get(otherIdx);
                StatsTuple nextSample;
                long additionalDelta = 0;
                if (selfSample.value < otherSample.value) {
                    nextSample = selfSample;
                    if (otherIdx > 0) {
                        additionalDelta = additionalSelfDelta;
                    }
                    selfIdx++;
                } else {
                    nextSample = otherSample;
                    if (selfIdx > 0) {
                        additionalDelta = additionalOtherDelta;
                    }
                    otherIdx++;
                }
                nextSample = nextSample.shallowCopy();
                nextSample.delta = nextSample.delta + additionalDelta;
                mergedSampled.add(nextSample);
            }
            IntStream.range(selfIdx, sampled.size())
                    .forEach(i -> mergedSampled.add(sampled.get(i)));
            IntStream.range(otherIdx, other.sampled.size())
                    .forEach(i -> mergedSampled.add(other.sampled.get(i)));

            List<StatsTuple> comp =
                    compressInternal(mergedSampled, 2 * mergedRelativeError * mergedCount);
            return new QuantileSummary(
                    mergedRelativeError, compressThreshold, comp, mergedCount, true);
        }
    }

    /**
     * Runs a query for a given percentile. The query can only be run on a compressed summary, you
     * need to call compress() before using it.
     *
     * @param percentile The target percentile.
     * @return The corresponding approximate quantile.
     */
    public double query(double percentile) {
        return query(new double[] {percentile})[0];
    }

    /**
     * Runs a query for a given sequence of percentiles. The query can only be run on a compressed
     * summary, you need to call compress() before using it.
     *
     * @param percentiles A list of the target percentiles.
     * @return A list of the corresponding approximate quantiles, in the same order as the input.
     */
    public double[] query(double[] percentiles) {
        Arrays.stream(percentiles)
                .forEach(
                        x ->
                                Preconditions.checkState(
                                        x >= 0 && x <= 1.0,
                                        "percentile should be in the range [0.0, 1.0]."));
        Preconditions.checkState(
                headBuffer.isEmpty(),
                "Cannot operate on an uncompressed summary, call compress() first.");
        Preconditions.checkState(
                sampled != null && !sampled.isEmpty(),
                "Cannot query percentiles without any records inserted.");
        double targetError = Long.MIN_VALUE;
        for (StatsTuple tuple : sampled) {
            targetError = Math.max(targetError, (tuple.delta + tuple.g));
        }
        targetError = targetError / 2;
        Map<Double, Integer> zipWithIndex = new HashMap<>(percentiles.length);
        IntStream.range(0, percentiles.length).forEach(i -> zipWithIndex.put(percentiles[i], i));

        int index = 0;
        long minRank = sampled.get(0).g;
        double[] sorted = Arrays.stream(percentiles).sorted().toArray();
        double[] result = new double[percentiles.length];

        for (double item : sorted) {
            int percentileIndex = zipWithIndex.get(item);
            if (item <= relativeError) {
                result[percentileIndex] = sampled.get(0).value;
            } else if (item >= 1 - relativeError) {
                result[percentileIndex] = sampled.get(sampled.size() - 1).value;
            } else {
                QueryResult queryResult =
                        findApproximateQuantile(index, minRank, targetError, item);
                index = queryResult.index;
                minRank = queryResult.minRankAtIndex;
                result[percentileIndex] = queryResult.percentile;
            }
        }
        return result;
    }

    /**
     * Checks whether the QuantileSummary has inserted rows. Running query on an empty
     * QuantileSummary would cause {@link java.lang.IllegalStateException}.
     *
     * @return True if the QuantileSummary is empty, otherwise false.
     */
    public boolean isEmpty() {
        return headBuffer.isEmpty() && sampled.isEmpty();
    }

    private QuantileSummary insertHeadBuffer() {
        if (headBuffer.isEmpty()) {
            return this;
        }

        long newCount = count;
        List<StatsTuple> newSamples = new ArrayList<>();
        List<Double> sorted = headBuffer.stream().sorted().collect(Collectors.toList());

        int cursor = 0;
        for (int i = 0; i < sorted.size(); i++) {
            while (cursor < sampled.size() && sampled.get(cursor).value <= sorted.get(i)) {
                newSamples.add(sampled.get(cursor));
                cursor++;
            }

            long delta = Double.valueOf(Math.floor(2.0 * relativeError * count)).longValue();
            if (newSamples.isEmpty() || (cursor == sampled.size() && i == sorted.size() - 1)) {
                delta = 0;
            }
            StatsTuple tuple = new StatsTuple(sorted.get(i), 1L, delta);
            newSamples.add(tuple);
            newCount++;
        }

        for (int i = cursor; i < sampled.size(); i++) {
            newSamples.add(sampled.get(i));
        }
        return new QuantileSummary(relativeError, compressThreshold, newSamples, newCount, false);
    }

    private List<StatsTuple> compressInternal(
            List<StatsTuple> currentSamples, double mergeThreshold) {
        if (currentSamples.isEmpty()) {
            return Collections.emptyList();
        }
        LinkedList<StatsTuple> result = new LinkedList<>();

        StatsTuple head = currentSamples.get(currentSamples.size() - 1);
        for (int i = currentSamples.size() - 2; i >= 1; i--) {
            StatsTuple tuple = currentSamples.get(i);
            if (tuple.g + head.g + head.delta < mergeThreshold) {
                head = head.shallowCopy();
                head.g = head.g + tuple.g;
            } else {
                result.addFirst(head);
                head = tuple;
            }
        }
        result.addFirst(head);

        StatsTuple currHead = currentSamples.get(0);
        if (currHead.value <= head.value && currentSamples.size() > 1) {
            result.addFirst(currHead);
        }
        return new ArrayList<>(result);
    }

    private QueryResult findApproximateQuantile(
            int index, long minRankAtIndex, double targetError, double percentile) {
        StatsTuple curSample = sampled.get(index);
        long rank = Double.valueOf(Math.ceil(percentile * count)).longValue();
        long minRank = minRankAtIndex;

        for (int i = index; i < sampled.size() - 1; ) {
            long maxRank = minRank + curSample.delta;
            if (maxRank - targetError < rank && rank <= minRank + targetError) {
                return new QueryResult(i, minRank, curSample.value);
            } else {
                curSample = sampled.get(++i);
                minRank += curSample.g;
            }
        }
        return new QueryResult(sampled.size() - 1, 0, sampled.get(sampled.size() - 1).value);
    }

    public double getRelativeError() {
        return relativeError;
    }

    public int getCompressThreshold() {
        return compressThreshold;
    }

    public long getCount() {
        return count;
    }

    public List<StatsTuple> getSampled() {
        return sampled;
    }

    public List<Double> getHeadBuffer() {
        return headBuffer;
    }

    public boolean isCompressed() {
        return compressed;
    }

    private QuantileSummary shallowCopy() {
        return new QuantileSummary(relativeError, compressThreshold, sampled, count, compressed);
    }

    /** Wrapper class to hold all information returned after querying. */
    private static class QueryResult {
        private final int index;
        private final long minRankAtIndex;
        private final double percentile;

        public QueryResult(int index, long minRankAtIndex, double percentile) {
            this.index = index;
            this.minRankAtIndex = minRankAtIndex;
            this.percentile = percentile;
        }
    }

    /**
     * Wrapper class to hold all statistics from the Greenwald-Khanna paper. It contains the
     * following information:
     *
     * <ul>
     *   <li>value: the sampled value.
     *   <li>g: the difference between the least rank of this element and the rank of the preceding
     *       element.
     *   <li>delta: the maximum span of the rank.
     * </ul>
     */
    public static class StatsTuple implements Serializable {
        private static final long serialVersionUID = 1L;
        public double value;
        public long g;
        public long delta;

        public StatsTuple() {}

        public StatsTuple(double value, long g, long delta) {
            this.value = value;
            this.g = g;
            this.delta = delta;
        }

        public StatsTuple shallowCopy() {
            return new StatsTuple(value, g, delta);
        }
    }
}
