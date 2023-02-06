################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import unittest
import numpy as np
import array

from pyflink.ml.linalg import SparseVector, DenseVector


class VectorTests(unittest.TestCase):
    def test_dot(self):
        sv = SparseVector(4, {1: 1, 3: 2})
        dv = DenseVector(np.array([1.0, 2.0, 3.0, 4.0]))
        lst = DenseVector([1, 2, 3, 4])
        mat = np.array(
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
        )
        arr = array.array("d", [0, 1, 2, 3])
        self.assertEqual(10.0, sv.dot(dv))
        self.assertTrue(np.array_equal(np.array([3.0, 6.0, 9.0, 12.0]), sv.dot(mat)))
        self.assertEqual(30.0, dv.dot(dv))
        self.assertTrue(np.array_equal(np.array([10.0, 20.0, 30.0, 40.0]), dv.dot(mat)))
        self.assertEqual(30.0, lst.dot(dv))
        self.assertTrue(np.array_equal(np.array([10.0, 20.0, 30.0, 40.0]), lst.dot(mat)))
        self.assertEqual(7.0, sv.dot(arr))

    def test_squared_distance(self):
        def squared_distance(a, b):
            if isinstance(a, (DenseVector, SparseVector)):
                return a.squared_distance(b)
            else:
                return b.squared_distance(a)

        sv = SparseVector(4, {1: 1, 3: 2})
        dv = DenseVector(np.array([1.0, 2.0, 3.0, 4.0]))
        lst = DenseVector([4, 3, 2, 1])
        lst1 = [4, 3, 2, 1]
        arr = array.array("d", [0, 2, 1, 3])
        narr = np.array([0, 2, 1, 3])
        self.assertEqual(15.0, squared_distance(sv, dv))
        self.assertEqual(25.0, squared_distance(sv, lst))
        self.assertEqual(20.0, squared_distance(dv, lst))
        self.assertEqual(15.0, squared_distance(dv, sv))
        self.assertEqual(25.0, squared_distance(lst, sv))
        self.assertEqual(20.0, squared_distance(lst, dv))
        self.assertEqual(0.0, squared_distance(sv, sv))
        self.assertEqual(0.0, squared_distance(dv, dv))
        self.assertEqual(0.0, squared_distance(lst, lst))
        self.assertEqual(25.0, squared_distance(sv, lst1))
        self.assertEqual(3.0, squared_distance(sv, arr))
        self.assertEqual(3.0, squared_distance(sv, narr))

    def test_eq(self):
        v1 = DenseVector([0.0, 1.0, 0.0, 5.5])
        v2 = SparseVector(4, [(1, 1.0), (3, 5.5)])
        v3 = DenseVector([0.0, 1.0, 0.0, 5.5])
        v4 = SparseVector(6, [(1, 1.0), (3, 5.5)])
        v5 = DenseVector([0.0, 1.0, 0.0, 2.5])
        v6 = SparseVector(4, [(1, 1.0), (3, 2.5)])
        self.assertEqual(v1, v2)
        self.assertEqual(v1, v3)
        self.assertFalse(v2 == v4)
        self.assertFalse(v1 == v5)
        self.assertFalse(v1 == v6)

    def test_get_set(self):
        v1 = DenseVector([0.0, 1.0, 0.0, 5.5])
        self.assertEqual(0.0, v1.get(0))
        v1.set(0, 1.0)
        self.assertEqual(1.0, v1.get(0))

        v2 = SparseVector(4, [(1, 1.0), (3, 5.5)])
        self.assertEqual(0.0, v2.get(0))

        v2.set(0, 1.0)
        self.assertEqual(1.0, v2.get(0))

        v2.set(1, 2.0)
        self.assertEqual(2.0, v2.get(1))
