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

package org.apache.flink.ml.common.ps.message;

/** Message type between workers and servers. */
public enum MessageType {
    /** The initialization request. */
    INITIALIZE(0),
    /** The pull request. */
    PUSH(1),
    /** The push request. */
    PULL(2),
    /** The all reduce request. */
    ALL_REDUCE(3);

    public final int type;

    MessageType(int type) {
        this.type = type;
    }

    public static MessageType valueOf(int value) {
        switch (value) {
            case 0:
                return MessageType.INITIALIZE;
            case 1:
                return MessageType.PUSH;
            case 2:
                return MessageType.PULL;
            case 3:
                return MessageType.ALL_REDUCE;
            default:
                throw new UnsupportedOperationException();
        }
    }
}
