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

package org.apache.flink.test.iteration.operators;

import org.apache.flink.api.common.functions.RichMapFunction;

/** Map Function triggers failover at the first task and first round. */
public class FailingMap<T> extends RichMapFunction<T, T> {

    private final int failingCount;

    private int count;

    public FailingMap(int failingCount) {
        this.failingCount = failingCount;
    }

    @Override
    public T map(T value) throws Exception {
        count++;
        if (getRuntimeContext().getIndexOfThisSubtask() == 0
                && getRuntimeContext().getAttemptNumber() == 0
                && count >= failingCount) {
            throw new RuntimeException("Artificial Exception");
        }

        return value;
    }
}
