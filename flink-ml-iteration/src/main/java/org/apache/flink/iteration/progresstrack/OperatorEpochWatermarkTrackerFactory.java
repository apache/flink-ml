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

package org.apache.flink.iteration.progresstrack;

import org.apache.flink.runtime.io.network.partition.consumer.InputGate;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.graph.StreamEdge;
import org.apache.flink.streaming.runtime.tasks.StreamTask;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * The factory of {@link OperatorEpochWatermarkTracker}. It analyze the inputs of an operator and
 * create the corresponding progress tracker.
 */
public class OperatorEpochWatermarkTrackerFactory {

    public static OperatorEpochWatermarkTracker create(
            StreamConfig streamConfig,
            StreamTask<?, ?> containingTask,
            OperatorEpochWatermarkTrackerListener progressTrackerListener) {

        int[] numberOfChannels;
        if (!streamConfig.isChainStart()) {
            numberOfChannels = new int[] {1};
        } else {
            InputGate[] inputGates = containingTask.getEnvironment().getAllInputGates();
            List<StreamEdge> inEdges =
                    streamConfig.getInPhysicalEdges(containingTask.getUserCodeClassLoader());

            // Mapping the edge type (input number) into a continuous sequence start from 0.
            TreeSet<Integer> edgeTypes = new TreeSet<>();
            inEdges.forEach(edge -> edgeTypes.add(edge.getTypeNumber()));

            Map<Integer, Integer> edgeTypeToIndices = new HashMap<>();
            for (int edgeType : edgeTypes) {
                edgeTypeToIndices.put(edgeType, edgeTypeToIndices.size());
            }

            numberOfChannels = new int[edgeTypeToIndices.size()];
            for (int i = 0; i < inEdges.size(); ++i) {
                numberOfChannels[edgeTypeToIndices.get(inEdges.get(i).getTypeNumber())] +=
                        inputGates[i].getNumberOfInputChannels();
            }
        }

        return new OperatorEpochWatermarkTracker(numberOfChannels, progressTrackerListener);
    }
}
