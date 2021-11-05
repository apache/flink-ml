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

package org.apache.flink.iteration.proxy.state;

import org.apache.flink.runtime.state.KeyedStateCheckpointOutputStream;
import org.apache.flink.runtime.state.OperatorStateCheckpointOutputStream;
import org.apache.flink.runtime.state.StateSnapshotContext;

/** Proxy {@link StateSnapshotContext} for the wrapped operators. */
public class ProxyStateSnapshotContext implements StateSnapshotContext {

    private final StateSnapshotContext wrappedContext;

    public ProxyStateSnapshotContext(StateSnapshotContext wrappedContext) {
        this.wrappedContext = wrappedContext;
    }

    @Override
    public KeyedStateCheckpointOutputStream getRawKeyedOperatorStateOutput() throws Exception {
        throw new UnsupportedOperationException(
                "Currently we do not support the raw operator state inside the iteration.");
    }

    @Override
    public OperatorStateCheckpointOutputStream getRawOperatorStateOutput() throws Exception {
        return wrappedContext.getRawOperatorStateOutput();
    }

    @Override
    public long getCheckpointId() {
        return wrappedContext.getCheckpointId();
    }

    @Override
    public long getCheckpointTimestamp() {
        return wrappedContext.getCheckpointTimestamp();
    }
}
