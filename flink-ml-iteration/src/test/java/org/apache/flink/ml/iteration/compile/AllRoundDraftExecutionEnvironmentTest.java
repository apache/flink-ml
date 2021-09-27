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

package org.apache.flink.ml.iteration.compile;

import org.apache.flink.ml.iteration.operator.OperatorWrapper;
import org.apache.flink.ml.iteration.operator.WrapperOperatorFactory;
import org.apache.flink.ml.iteration.operator.allround.AllRoundOperatorWrapper;
import org.apache.flink.ml.iteration.typeinfo.IterationRecordSerializer;
import org.apache.flink.streaming.api.graph.StreamEdge;
import org.apache.flink.streaming.api.graph.StreamGraph;
import org.apache.flink.streaming.api.graph.StreamNode;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class AllRoundDraftExecutionEnvironmentTest extends DraftExecutionEnvironmentTestBase {

    @Override
    protected OperatorWrapper<?, ?> getOperatorWrapper() {
        return new AllRoundOperatorWrapper<>();
    }

    @Override
    protected void checkWrappedGraph(
            StreamGraph draftStreamGraph,
            StreamGraph actualStreamGraph,
            DraftExecutionEnvironment draftEnv) {
        assertNotNull("Draft stream graph should not be null", draftStreamGraph);
        assertNotNull("Actual stream graph should not be null", actualStreamGraph);

        // First check the two graphs has the isomorphic structure.
        assertEquals(
                draftStreamGraph.getStreamNodes().size(),
                actualStreamGraph.getStreamNodes().size());
        for (StreamNode draftNode : draftStreamGraph.getStreamNodes()) {
            StreamNode actualNode =
                    actualStreamGraph.getStreamNode(
                            draftEnv.getActualStream(draftNode.getId()).getId());
            assertEquals(
                    getSortedOutEdgeTarget(draftNode).stream()
                            .map(id -> draftEnv.getActualStream(id).getId())
                            .collect(Collectors.toList()),
                    getSortedOutEdgeTarget(actualNode));
        }

        // Now check each operator is correctly wrapped.
        for (StreamNode draftNode : draftStreamGraph.getStreamNodes()) {
            StreamNode actualNode =
                    actualStreamGraph.getStreamNode(
                            draftEnv.getActualStream(draftNode.getId()).getId());

            // Not check sources.
            if (actualNode.getInEdges().size() == 0) {
                continue;
            }

            assertEquals(WrapperOperatorFactory.class, actualNode.getOperatorFactory().getClass());
            assertEquals(
                    AllRoundOperatorWrapper.class,
                    ((WrapperOperatorFactory) actualNode.getOperatorFactory())
                            .getWrapper()
                            .getClass());
            assertEquals(
                    draftNode.getOperatorFactory().getClass(),
                    ((WrapperOperatorFactory) actualNode.getOperatorFactory())
                            .getOperatorFactory()
                            .getClass());
            assertEquals(
                    draftNode
                            .getOperatorFactory()
                            .getStreamOperatorClass(getClass().getClassLoader()),
                    ((WrapperOperatorFactory) actualNode.getOperatorFactory())
                            .getOperatorFactory()
                            .getStreamOperatorClass(getClass().getClassLoader()));

            if (actualNode.getTypeSerializerOut() != null) {
                assertEquals(
                        IterationRecordSerializer.class,
                        actualNode.getTypeSerializerOut().getClass());
                assertEquals(
                        draftNode.getTypeSerializerOut().getClass(),
                        ((IterationRecordSerializer) actualNode.getTypeSerializerOut())
                                .getInnerSerializer()
                                .getClass());
            }

            assertEquals(draftNode.getStateKeySerializer(), actualNode.getStateKeySerializer());
            assertEquals(draftNode.getParallelism(), actualNode.getParallelism());
            assertEquals(draftNode.getBufferTimeout(), actualNode.getBufferTimeout());
            assertEquals(draftNode.getCoLocationGroup(), actualNode.getCoLocationGroup());
            assertEquals(draftNode.getUserHash(), actualNode.getUserHash());
            assertEquals(
                    draftNode.getManagedMemorySlotScopeUseCases(),
                    actualNode.getManagedMemorySlotScopeUseCases());
            assertEquals(
                    draftNode.getManagedMemoryOperatorScopeUseCaseWeights(),
                    actualNode.getManagedMemoryOperatorScopeUseCaseWeights());
        }
    }

    private List<Integer> getSortedOutEdgeTarget(StreamNode node) {
        return node.getOutEdges().stream()
                .map(StreamEdge::getTargetId)
                .sorted()
                .collect(Collectors.toList());
    }
}
