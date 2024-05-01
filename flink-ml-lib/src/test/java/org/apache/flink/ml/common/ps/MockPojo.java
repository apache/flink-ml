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

package org.apache.flink.ml.common.ps;

/** Mock pojo class to test all reduce. */
public class MockPojo {
    public int i;
    public int j;

    public MockPojo(int i, int j) {
        this.i = i;
        this.j = j;
    }

    public MockPojo() {}

    @Override
    public String toString() {
        return i + "-" + j;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof MockPojo) {
            MockPojo other = (MockPojo) obj;
            return i == other.i && j == other.j;
        }
        return false;
    }
}
