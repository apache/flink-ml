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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.ml.common.ps.iterations.MLSessionImpl;
import org.apache.flink.ml.common.ps.sarray.SharedDoubleArray;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;
import org.apache.flink.ml.recommendation.als.Als.Ratings;

import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/** The ML session for als training with PS. */
public class AlsMLSession extends MLSessionImpl<Ratings> {

    /** Indices for pulling the data. */
    public SharedLongArray pullIndices = new SharedLongArray();

    /** Values for pulling the data. */
    public SharedDoubleArray pullValues = new SharedDoubleArray();

    /** Indices for pushing the data. */
    public SharedLongArray pushIndices = new SharedLongArray();

    /** Values for pushing the data. */
    public SharedDoubleArray pushValues = new SharedDoubleArray();

    /** The all reduce buffer for computing yty. */
    public double[][] allReduceBuffer;

    /** The aggregator array for computing yty. */
    public SharedDoubleArray aggregatorSDAArray;

    /** Ratings data for current iteration. */
    public BlockData batchData;

    /** The intermediate variable for updating factors. */
    public double[] yty;

    /** Ratings batch data list for user. */
    private List<BlockData> userRatingsList;

    /** Ratings batch data list for item. */
    private List<BlockData> itemRatingsList;

    /** Num blocks of user ratings data. */
    public int numUserBlocks;

    /** Num blocks of item ratings data. */
    public int numItemBlocks;

    /** Processing user batch index in current iteration. */
    public int currentUserIndex;

    /** Processing item batch index in current iteration. */
    public int currentItemIndex;

    /** Current iteration, updates user factors or item factors. */
    public boolean updateUserFactors = true;

    /** Initialized Rating or not. */
    public boolean isRatingsInitialized = false;

    public static final Logger LOG = LoggerFactory.getLogger(Als.class);

    public LongOpenHashSet reusedNeighborsSet;
    public Long2IntOpenHashMap reusedNeighborIndexMapping;

    private final int parallelism;
    private long timing = 0;
    private final boolean implicit;

    public long[] userIds;
    public long[] itemIds;

    private int maxNumNeighbors = 0;
    private int maxNumNodes = 0;
    private final int rank;

    public AlsMLSession(boolean implicit, int rank, int parallelism) {
        this.rank = rank;
        this.parallelism = parallelism;
        this.implicit = implicit;
        if (implicit) {
            aggregatorSDAArray = new SharedDoubleArray(new double[rank * rank]);
            allReduceBuffer = new double[][] {new double[rank * rank]};
        }
    }

    @Override
    public void setWorldInfo(int workerId, int numWorkers) {
        this.workerId = workerId;
        this.numWorkers = numWorkers;
    }

    /** Initializes ratings data in minibatch list format. */
    public void initializeRatingsBatchData() throws IOException {
        final int taskCapacity = 8 * 1024 * 1024;
        final int defaultNumBlocks = 1;

        long numUser = 0L;
        long numItem = 0L;
        while (inputData.hasNext()) {
            long numSamples;
            Ratings ratings = inputData.next();
            if (ratings.scores == null) {
                numUser = ratings.neighbors[0];
                numItem = ratings.neighbors[1];
                numSamples = ratings.neighbors[2];
                long hottestUserPoint = ratings.neighbors[3];
                long hottestItemPoint = ratings.neighbors[4];

                if (numItem * rank < taskCapacity) {
                    this.numUserBlocks = defaultNumBlocks;
                } else {
                    this.numUserBlocks =
                            (int) (numSamples * rank / (parallelism * taskCapacity)) + 1;
                }
                if (numUser * rank < taskCapacity) {
                    this.numItemBlocks = defaultNumBlocks;
                } else {
                    this.numItemBlocks =
                            (int) (numSamples * rank / (parallelism * taskCapacity)) + 1;
                }

                LOG.info("rank : " + rank);

                LOG.info("num total users : " + numUser);
                LOG.info("num total items : " + numItem);
                LOG.info("num total samples : " + numSamples);

                LOG.info("num user blocks : " + numUserBlocks);
                LOG.info("num item blocks : " + numItemBlocks);

                LOG.info("hottest user point : " + hottestUserPoint);
                LOG.info("hottest item point : " + hottestItemPoint);
                break;
            }
        }

        this.userRatingsList = new ArrayList<>(numUserBlocks);
        int userBlockSize = (int) numUser / (numUserBlocks * parallelism);
        for (int i = 0; i < numUserBlocks; ++i) {
            BlockData blockData = new BlockData(new ArrayList<>(userBlockSize), false);
            this.userRatingsList.add(blockData);
        }
        this.itemRatingsList = new ArrayList<>(numItemBlocks);
        int itemBlockSize = (int) numItem / (numItemBlocks * parallelism);
        for (int i = 0; i < numItemBlocks; ++i) {
            BlockData blockData = new BlockData(new ArrayList<>(itemBlockSize), false);
            this.itemRatingsList.add(blockData);
        }

        inputData.reset();

        while (inputData.hasNext()) {
            Ratings ratings = inputData.next();
            if (ratings.scores == null) {
                continue;
            }
            if (ratings.nodeId % 2 == 0) {
                int blockId = (int) (ratings.nodeId / 2) % numUserBlocks;
                this.userRatingsList.get(blockId).add(ratings);
                if (!this.userRatingsList.get(blockId).hasHotPoint) {
                    this.userRatingsList.get(blockId).hasHotPoint = ratings.isSplit;
                }
            } else {
                int blockId = (int) (ratings.nodeId / 2) % numItemBlocks;
                this.itemRatingsList.get(blockId).add(ratings);
                if (!this.itemRatingsList.get(blockId).hasHotPoint) {
                    this.itemRatingsList.get(blockId).hasHotPoint = ratings.isSplit;
                }
            }
        }

        for (BlockData blockData : userRatingsList) {
            initializeBlockData(blockData);
        }

        for (BlockData blockData : itemRatingsList) {
            initializeBlockData(blockData);
        }

        pullIndices.size(maxNumNeighbors);
        pullValues.size(maxNumNeighbors * rank);
        pushIndices.size(maxNumNodes);
        pushValues.size(maxNumNodes * rank);
        this.reusedNeighborIndexMapping = new Long2IntOpenHashMap(maxNumNeighbors);
        this.reusedNeighborsSet = new LongOpenHashSet(maxNumNeighbors);

        if (this.implicit) {
            LongOpenHashSet longOpenHashSet = new LongOpenHashSet();
            for (BlockData blockData : itemRatingsList) {
                for (Ratings r : blockData.ratingsList) {
                    if (r.isMainNode && r.isSplit) {
                        longOpenHashSet.add(r.nodeId);
                    } else if (!r.isSplit) {
                        longOpenHashSet.add(r.nodeId);
                    }
                }
            }
            itemIds = new long[longOpenHashSet.size()];
            Iterator<Long> iterator = longOpenHashSet.iterator();
            int it = 0;
            while (iterator.hasNext()) {
                itemIds[it++] = iterator.next();
            }
            longOpenHashSet.clear();
            for (BlockData blockData : userRatingsList) {
                for (Ratings r : blockData.ratingsList) {
                    if (r.isMainNode && r.isSplit) {
                        longOpenHashSet.add(r.nodeId);
                    } else if (!r.isSplit) {
                        longOpenHashSet.add(r.nodeId);
                    }
                }
            }
            userIds = new long[longOpenHashSet.size()];
            iterator = longOpenHashSet.iterator();
            it = 0;
            while (iterator.hasNext()) {
                userIds[it++] = iterator.next();
            }
        }
    }

    private void initializeBlockData(BlockData blockData) {
        LongOpenHashSet neighborsSet = new LongOpenHashSet(blockData.ratingsList.size());

        for (Ratings dataPoint : blockData.ratingsList) {
            for (long index : dataPoint.neighbors) {
                neighborsSet.add(index);
            }
            if (!dataPoint.isSplit) {
                blockData.numCommonNodeIds++;
            } else {
                if (dataPoint.isMainNode) {
                    blockData.numSplitNodeIds++;
                }
            }
        }
        maxNumNeighbors = Math.max(maxNumNeighbors, neighborsSet.size());
        maxNumNodes = Math.max(maxNumNodes, blockData.numCommonNodeIds);
    }

    public void prepareNextRatingsBatchData() throws IOException {
        if (!isRatingsInitialized) {
            initializeRatingsBatchData();
            isRatingsInitialized = true;
        }

        if (updateUserFactors) {
            this.batchData = userRatingsList.get(currentUserIndex++);
            if (currentUserIndex == numUserBlocks) {
                currentUserIndex = 0;
                updateUserFactors = false;
            }
        } else {
            this.batchData = itemRatingsList.get(currentItemIndex++);
            if (currentItemIndex == numItemBlocks) {
                currentItemIndex = 0;
                updateUserFactors = true;
            }
        }
    }

    private long clock() {
        long current = System.currentTimeMillis();
        long duration = current - timing;
        timing = current;
        return duration;
    }

    public void log(String className, boolean start) {
        if (start) {
            LOG.info(
                    String.format(
                            "[Worker-%d, iteration-%d] starts %s, %d%n",
                            workerId, iterationId, className, clock()));
        } else {
            LOG.info(
                    String.format(
                            "[Worker-%d, iteration-%d] ends %s, %d%n",
                            workerId, iterationId, className, clock()));
        }
    }

    /** The computing block data in every iteration. */
    public static class BlockData {
        public BlockData(List<Ratings> ratingsList, boolean hasHotPoint) {
            this.ratingsList = ratingsList;
            this.hasHotPoint = hasHotPoint;
        }

        public List<Ratings> ratingsList;

        public boolean hasHotPoint;

        public int numCommonNodeIds;

        public int numSplitNodeIds;

        public Ratings get(int idx) {
            return ratingsList.get(idx);
        }

        public void add(Ratings ratings) {
            ratingsList.add(ratings);
        }
    }
}
