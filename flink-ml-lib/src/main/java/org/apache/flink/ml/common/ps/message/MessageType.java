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

/** Message Type between workers and servers. */
public enum MessageType {
    /** Message sent from workers to servers, which initializes the model on servers as zero. */
    INITIALIZE_MODEL_AS_ZERO((char) 0),
    /** Message sent from workers to servers, which specifies the indices of model to pull. */
    PULL_INDEX((char) 1),
    /**
     * Message sent from server to workers, which specifies the values of the model pulled from
     * servers.
     */
    PULLED_VALUE((char) 2),
    /**
     * Message sent from workers to servers, which specifies the indices and values of the model to
     * push to servers.
     */
    PUSH_KV((char) 3);

    public final char type;

    MessageType(char type) {
        this.type = type;
    }

    public static MessageType valueOf(char value) {
        switch (value) {
            case (char) 0:
                return MessageType.INITIALIZE_MODEL_AS_ZERO;
            case (char) 1:
                return MessageType.PULL_INDEX;
            case ((char) 2):
                return MessageType.PULLED_VALUE;
            case ((char) 3):
                return MessageType.PUSH_KV;
            default:
                throw new UnsupportedOperationException();
        }
    }
}
