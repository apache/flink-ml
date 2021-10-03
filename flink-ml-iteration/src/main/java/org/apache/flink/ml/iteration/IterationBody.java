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

package org.apache.flink.ml.iteration;

/** The builder of the subgraph that will be executed inside the iteration. */
public interface IterationBody {

    /**
     * This method creates the graph for the iteration body. See {@link Iterations} for how the
     * iteration body can be executed and terminated.
     *
     * @param variableStreams the variable streams, which will be updated via a feedback stream in
     *     each round.
     * @param dataStreams the streams referred in the iteration body, which will only be emitted in
     *     the first round.
     * @return the result of the iteration, including the feedbacks and outputs.
     */
    IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams);
}
