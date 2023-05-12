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
    ZEROS_TO_PUSH((char) 0),
    INDICES_TO_PULL((char) 1),
    VALUES_PULLED((char) 2),
    KVS_TO_PUSH((char) 3);

    public final char type;

    MessageType(char type) {
        this.type = type;
    }

    public static MessageType valueOf(char value) {
        switch (value) {
            case (char) 0:
                return MessageType.ZEROS_TO_PUSH;
            case (char) 1:
                return MessageType.INDICES_TO_PULL;
            case ((char) 2):
                return MessageType.VALUES_PULLED;
            case ((char) 3):
                return MessageType.KVS_TO_PUSH;
            default:
                throw new UnsupportedOperationException();
        }
    }
}
