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

import org.apache.flink.annotation.Experimental;
import org.apache.flink.iteration.IterationListener;

import java.io.Serializable;

/**
 * A read request for a shared object with given step offset. The step {@link OFFSET} is used to
 * calculate read-step from current operator step.
 *
 * <p>The concept of `step` is first defined on operators. Every operator maintains its `step`
 * implicitly. For operators in non-iterations usage, their `step`s are treated as constants. While
 * for operators in iterations usage, their `step`s are bound to the epoch watermarks:
 *
 * <p>With every call of {@link IterationListener#onEpochWatermarkIncremented}, the value of step is
 * set to the epoch watermark. Before the first call of {@link
 * IterationListener#onEpochWatermarkIncremented}, the step is set to a small enough value. While
 * after {@link IterationListener#onIterationTerminated}, the step is set to a large enough value.
 * In this way, the changes of step can be considered as an ordered sequence. Note that, the `step`
 * is implicitly maintained by the infrastructure, even if the operator is not implementing {@link
 * IterationListener}.
 *
 * <p>Then, the concept of `step` is defined on reads and writes of shared objects. Every write
 * brings the step of its owner operator at that moment, which is named as `write-step`. To read the
 * shared object with the exact `write-step`, the reader operator must provide a same `read-step`.
 * The `read-step` could be different from that of the reader operator, and their difference is kept
 * unchanged, which is the step offset defined in {@link ReadRequest#offset}.
 *
 * @param <T> The type of the shared object.
 */
@Experimental
public class ReadRequest<T> implements Serializable {
    final Descriptor<T> descriptor;
    final OFFSET offset;

    ReadRequest(Descriptor<T> descriptor, OFFSET offset) {
        this.descriptor = descriptor;
        this.offset = offset;
    }

    enum OFFSET {
        SAME,
        PREV,
        NEXT,
    }
}
