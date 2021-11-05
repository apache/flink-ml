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

package org.apache.flink.iteration.operator.event;

import org.apache.flink.iteration.operator.coordinator.HeadOperatorCoordinator;
import org.apache.flink.runtime.operators.coordination.OperatorEvent;

/**
 * Notifies the {@link HeadOperatorCoordinator} that the status has been terminating on startup so
 * that the coordinator would not emit {@link CoordinatorCheckpointEvent}.
 */
public class TerminatingOnInitializeEvent implements OperatorEvent {

    public static final TerminatingOnInitializeEvent INSTANCE = new TerminatingOnInitializeEvent();

    private TerminatingOnInitializeEvent() {}
}
