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

package org.apache.flink.ml.common.sharedobjects;

import org.apache.flink.streaming.api.operators.AbstractStreamOperator;

import java.util.UUID;

/**
 * A base class of stream operators where shared objects are required.
 *
 * <p>Official subclasses, i.e., {@link AbstractSharedObjectsOneInputStreamOperator} and {@link
 * AbstractSharedObjectsTwoInputStreamOperator}, are strongly recommended.
 *
 * <p>If you are going to implement a subclass by yourself, you have to handle potential deadlocks.
 */
public abstract class AbstractSharedObjectsStreamOperator<OUT> extends AbstractStreamOperator<OUT> {

    /**
     * A unique identifier for the instance, which is kept unchanged between client side and
     * runtime.
     */
    private final String accessorID;

    /** The context for shared objects reads/writes. */
    protected transient SharedObjectsContext context;

    AbstractSharedObjectsStreamOperator() {
        super();
        accessorID = getClass().getSimpleName() + "-" + UUID.randomUUID();
    }

    void onSharedObjectsContextSet(SharedObjectsContext context) {
        this.context = context;
    }

    String getAccessorID() {
        return accessorID;
    }
}
