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

/**
 * The message to be passed between worker node and server node.
 *
 * <p>NOTE: Every Message subclass should implement a static method with signature {@code static T
 * fromBytes(byte[] bytesData)}, where {@code T} refers to the concrete subclass. This static method
 * should instantiate a new Message instance based on the data read from the given byte array.
 */
public interface Message {
    /**
     * Serializes the message into a byte array.
     *
     * <p>Note that the first two bytes of the result buffer is reserved for {@link MessageType}.
     */
    byte[] toBytes();
}
