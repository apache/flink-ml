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

package org.apache.flink.iteration.operator;

import org.apache.flink.api.common.state.ListState;

import java.util.Iterator;
import java.util.Optional;

import static org.apache.flink.util.Preconditions.checkState;

/** Utility to deal with the states inside the operator. */
public class OperatorStateUtils {

    public static <T> Optional<T> getUniqueElement(ListState<T> listState, String stateName)
            throws Exception {
        Iterator<T> iterator = listState.get().iterator();
        if (!iterator.hasNext()) {
            return Optional.empty();
        }

        T result = iterator.next();
        checkState(!iterator.hasNext(), "The state " + stateName + " has more that one elements");
        return Optional.of(result);
    }
}
