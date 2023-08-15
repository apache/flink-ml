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

package org.apache.flink.iteration.datacache.nonkeyed;

import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Assert;
import org.junit.Test;

/** Tests the behavior of the {@link OperatorScopeManagedMemoryManager}. */
public class OperatorScopeManagedMemoryManagerTest {

    private static final double EPS = 1e-9;

    @Test
    public void testUsage() {
        OperatorScopeManagedMemoryManager manager = new OperatorScopeManagedMemoryManager();
        manager.register("state-1", 100);
        manager.register("state-2", 400);
        Assert.assertEquals(manager.getFraction("state-1"), 0.2, EPS);
        Assert.assertEquals(manager.getFraction("state-2"), 0.8, EPS);
    }

    @Test
    public void testZeroUsage() {
        OperatorScopeManagedMemoryManager manager = new OperatorScopeManagedMemoryManager();
        manager.register("state-1", 0);
        manager.register("state-2", 0);
        Assert.assertEquals(manager.getFraction("state-1"), 0, EPS);
        Assert.assertEquals(manager.getFraction("state-2"), 0, EPS);
    }

    @Test
    public void testInvalidUsage() {
        OperatorScopeManagedMemoryManager manager = new OperatorScopeManagedMemoryManager();
        try {
            manager.register("state-1", 100);
            Assert.assertEquals(manager.getFraction("state-1"), 1., EPS);
            manager.register("state-2", 400);
            Assert.assertEquals(manager.getFraction("state-2"), 0.8, EPS);
            throw new RuntimeException();
        } catch (Exception e) {
            Assert.assertEquals(
                    IllegalStateException.class, ExceptionUtils.getRootCause(e).getClass());
        }
    }
}
