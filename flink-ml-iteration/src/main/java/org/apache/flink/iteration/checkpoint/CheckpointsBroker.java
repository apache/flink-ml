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

package org.apache.flink.iteration.checkpoint;

import org.apache.flink.statefun.flink.core.feedback.SubtaskFeedbackKey;

import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Hand offs the {@link Checkpoints} from the head operator to the tail operator so that the tail
 * operator could decrease the reference count of the raw state when checkpoints are aborted. We
 * could not count on the head operator since it would be blocked on closing the raw state when
 * aborting the checkpoint. It also looks like a bug.
 */
public class CheckpointsBroker {

    private static final CheckpointsBroker INSTANCE = new CheckpointsBroker();

    private final ConcurrentHashMap<SubtaskFeedbackKey<?>, Checkpoints<?>> checkpointManagers =
            new ConcurrentHashMap<>();

    public static CheckpointsBroker get() {
        return INSTANCE;
    }

    public <V> void setCheckpoints(SubtaskFeedbackKey<V> key, Checkpoints<V> checkpoints) {
        checkpointManagers.put(key, checkpoints);
    }

    @SuppressWarnings({"unchecked"})
    public <V> Checkpoints<V> getCheckpoints(SubtaskFeedbackKey<V> key) {
        Objects.requireNonNull(key);
        return (Checkpoints<V>) Objects.requireNonNull(checkpointManagers.get(key));
    }

    @SuppressWarnings("resource")
    void removeChannel(SubtaskFeedbackKey<?> key) {
        checkpointManagers.remove(key);
    }
}
