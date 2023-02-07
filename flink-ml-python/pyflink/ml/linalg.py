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
import array
import struct
from abc import ABC, abstractmethod
from typing import Union, Sized

import numpy as np
from pyflink.common import TypeInformation
from pyflink.fn_execution.stream_slow import OutputStream, InputStream
from pyflink.java_gateway import get_gateway


class VectorTypeInfo(TypeInformation):
    def __init__(self):
        super(VectorTypeInfo, self).__init__()
        self._dense_vector_type_info = DenseVectorTypeInfo()
        self._sparse_vector_type_info = SparseVectorTypeInfo()

    def need_conversion(self):
        return True

    def to_internal_type(self, obj):
        if obj is None:
            return None
        if isinstance(obj, DenseVector):
            return chr(0).encode('latin-1') + self._dense_vector_type_info.to_internal_type(obj)
        else:
            return chr(1).encode('latin-1') + self._sparse_vector_type_info.to_internal_type(obj)

    def from_internal_type(self, obj):
        if obj is not None:
            if obj[0] == 0:
                return self._dense_vector_type_info.from_internal_type(obj[1:])
            else:
                return self._sparse_vector_type_info.from_internal_type(obj[1:])

    def get_java_type_info(self):
        if not self._j_typeinfo:
            self._j_typeinfo = get_gateway().jvm \
                .org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo.INSTANCE

        return self._j_typeinfo

    def __eq__(self, o: object) -> bool:
        return isinstance(o, VectorTypeInfo)

    def __repr__(self):
        return "VectorTypeInfo"


class DenseVectorTypeInfo(TypeInformation):
    def __init__(self):
        super(DenseVectorTypeInfo, self).__init__()
        self._output_stream = OutputStream()
        self._input_stream = InputStream(None)

    def need_conversion(self):
        return True

    def to_internal_type(self, obj):
        if obj is None:
            return
        assert isinstance(obj, DenseVector)
        values = [float(v) for v in obj._values]
        stream = self._output_stream
        stream.write_int32(len(values))
        for value in values:
            stream.write_double(value)
        internal_data = bytearray(stream.get())
        stream.clear()
        return internal_data

    def from_internal_type(self, obj):
        if obj is not None:
            assert isinstance(obj, bytearray)
            # reset input stream
            stream = self._input_stream
            stream.data = bytes(obj)
            stream.pos = 0

            length = stream.read_int32()
            values = [stream.read_double() for _ in range(length)]
            return Vectors.dense(values)

    def get_java_type_info(self):
        if not self._j_typeinfo:
            self._j_typeinfo = get_gateway().jvm \
                .org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo.INSTANCE
        return self._j_typeinfo

    def __eq__(self, o: object) -> bool:
        return isinstance(o, DenseVectorTypeInfo)

    def __repr__(self):
        return "DenseVectorTypeInfo"


class SparseVectorTypeInfo(TypeInformation):
    def __init__(self):
        super(SparseVectorTypeInfo, self).__init__()
        self._output_stream = OutputStream()
        self._input_stream = InputStream(None)

    def need_conversion(self):
        return True

    def to_internal_type(self, obj):
        if obj is None:
            return
        assert isinstance(obj, SparseVector)
        stream = self._output_stream
        stream.write_int32(obj.size())

        l = len(obj._values)
        stream.write_int32(l)
        for i in range(l):
            stream.write_int32(obj._indices[i])
            stream.write_double(obj._values[i])
        internal_data = bytearray(stream.get())
        stream.clear()
        return internal_data

    def from_internal_type(self, obj):
        if obj is not None:
            assert isinstance(obj, bytearray)
            # reset input stream
            stream = self._input_stream
            stream.data = bytes(obj)
            stream.pos = 0

            size = stream.read_int32()
            values = {}
            for i in range(stream.read_int32()):
                k = stream.read_int32()
                v = stream.read_double()
                values[k] = v
            return Vectors.sparse(size, values)

    def get_java_type_info(self):
        if not self._j_typeinfo:
            self._j_typeinfo = get_gateway().jvm \
                .org.apache.flink.ml.linalg.typeinfo.SparseVectorTypeInfo.INSTANCE
        return self._j_typeinfo

    def __eq__(self, o) -> bool:
        return isinstance(o, SparseVectorTypeInfo)

    def __repr__(self):
        return "SparseVectorTypeInfo"


class DenseMatrixTypeInfo(TypeInformation):
    def __init__(self):
        super(DenseMatrixTypeInfo, self).__init__()
        self._output_stream = OutputStream()
        self._input_stream = InputStream(None)

    def need_conversion(self):
        return True

    def to_internal_type(self, matrix):
        if matrix is None:
            return
        assert isinstance(matrix, DenseMatrix)
        stream = self._output_stream
        stream.write_int32(matrix.num_rows())
        stream.write_int32(matrix.num_cols())
        for value in matrix._values:
            stream.write_double(value)

    def from_internal_type(self, obj):
        if obj is not None:
            assert isinstance(obj, bytearray)
            # reset input stream
            stream = self._input_stream
            stream.data = bytes(obj)
            stream.pos = 0

            m = stream.read_int32()
            n = stream.read_int32()
            values = [stream.read_double() for _ in range(m * n)]
            return DenseMatrix(m, n, values)

    def get_java_type_info(self):
        if not self._j_typeinfo:
            self._j_typeinfo = get_gateway().jvm \
                .org.apache.flink.ml.linalg.typeinfo.DenseMatrixTypeInfo.INSTANCE
        return self._j_typeinfo

    def __eq__(self, o: object) -> bool:
        return isinstance(o, DenseMatrixTypeInfo)

    def __repr__(self):
        return "DenseMatrixTypeInfo"


class Vector(ABC):
    """
    Abstract class for DenseVector and SparseVector.
    """

    @abstractmethod
    def size(self) -> int:
        """
        Gets the size of the vector.
        """
        pass

    @abstractmethod
    def set(self, i: int, value: np.float64):
        """
        Sets the value of the ith element.
        """
        pass

    @abstractmethod
    def get(self, i: int):
        """
        Gets the value of the ith element.
        """
        pass

    @abstractmethod
    def to_array(self) -> np.ndarray:
        """
        Convert the vector into an numpy.ndarray
        """
        pass

    @staticmethod
    def _equals(v1_indices, v1_values, v2_indices, v2_values):
        v1_size = len(v1_values)
        v2_size = len(v2_values)
        k1 = 0
        k2 = 0
        all_equal = True
        while all_equal:
            while k1 < v1_size and v1_values[k1] == 0:
                k1 += 1
            while k2 < v2_size and v2_values[k2] == 0:
                k2 += 1

            if k1 >= v1_size or k2 >= v2_size:
                return k1 >= v1_size and k2 >= v2_size

            all_equal = v1_indices[k1] == v2_indices[k2] and v1_values[k1] == v2_values[k2]
            k1 += 1
            k2 += 1
        return all_equal

    def __len__(self):
        return self.size()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get(key)
        raise TypeError("Invalid argument type")


class DenseVector(Vector):
    """
    A dense vector represented by a value array. We use numpy array for storage and arithmetic
    will be delegated to the underlying numpy array.

    Examples:
    ::

        >>> v = Vectors.dense([1.0, 2.0])
        >>> u = Vectors.dense([3.0, 4.0])
        >>> v + u
        DenseVector([4.0, 6.0])
        >>> 2 - v
        DenseVector([1.0, 0.0])
        >>> v / 2
        DenseVector([0.5, 1.0])
        >>> v * u
        DenseVector([3.0, 8.0])
        >>> u / v
        DenseVector([3.0, 2.0])
        >>> u % 2
        DenseVector([1.0, 0.0])
        >>> -v
        DenseVector([-1.0, -2.0])
    """

    def __init__(self, values):
        if not isinstance(values, np.ndarray):
            values = np.array(values, dtype=np.float64)
        if values.dtype != np.float64:
            values = values.astype(np.float64)
        self._values = values

    def size(self) -> int:
        return len(self._values)

    def get(self, i: int):
        return self._values[i]

    def set(self, i: int, value: np.float64):
        self._values[i] = value

    def to_array(self) -> np.ndarray:
        return self._values

    def dot(self, other: Union[Vector, np.ndarray, Sized]) -> np.ndarray:
        """
        Dot product of two Vectors.

        Examples:
        ::

            >>> import  array
            >>> dense = DenseVector(array.array('d', [1., 2.]))
            >>> dense.dot(dense)
            5.0
            >>> dense.dot(SparseVector(2, [0, 1], [2., 1.]))
            4.0
            >>> dense.dot(range(1, 3))
            5.0
            >>> dense.dot(np.array(range(1, 3)))
            5.0
            >>> dense.dot(np.reshape([1., 2., 3., 4.], (2, 2), order='F'))
            array([ 5., 11.])
        """
        if type(other) == np.ndarray:
            return np.dot(self._values, other)
        else:
            assert len(self) == len(other), "dimension mismatch"
            if isinstance(other, SparseVector):
                return other.dot(self)
            elif isinstance(other, Vector):
                return np.dot(self._values, other.to_array())
            else:
                return np.dot(self._values, other)

    def squared_distance(self, other: Union[Vector, np.ndarray, Sized]) -> np.ndarray:
        """
        Squared distance of two Vectors.

        Examples:
        ::

            >>> import array
            >>> dense1 = DenseVector(array.array('d', [1., 2.]))
            >>> dense1.squared_distance(dense1)
            0.0
            >>> dense2 = np.array((2., 1.))
            >>> dense1.squared_distance(dense2)
            2.0
            >>> sparse1 = SparseVector(2, [0, 1], [2., 1.])
            >>> dense1.squared_distance(sparse1)
            2.0
        """
        assert len(self) == len(other), "dimension mismatch"
        if isinstance(other, SparseVector):
            return other.squared_distance(self)

        if isinstance(other, Vector):
            other = other.to_array()
        elif not isinstance(other, np.ndarray):
            other = np.array(other)
        diff = self._values - other
        return np.dot(diff, diff)

    @property
    def values(self):
        """
        Returns the underlying numpy.ndarray
        """
        return self._values

    def __getitem__(self, item):
        return self._values[item]

    def __len__(self):
        return self.size()

    def __eq__(self, other):
        if isinstance(other, DenseVector):
            return np.array_equal(self._values, other._values)
        elif isinstance(other, SparseVector):
            if self.size() != other.size():
                return False
            return Vector._equals(
                list(range(len(self))), self._values, other._indices, other._values)
        return False

    def __str__(self):
        return "[" + ",".join([str(v) for v in self._values]) + "]"

    def __repr__(self):
        return "DenseVector([%s])" % (", ".join(str(i) for i in self._values))

    def __hash__(self):
        size = len(self)
        result = 31 + size
        nnz = 0
        i = 0
        while i < size and nnz < 128:
            if self._values[i] != 0:
                result = 31 * result + i
                bits = _double_to_long_bits(self._values[i])
                result = 31 * result + (bits ^ (bits >> 32))
                nnz += 1
            i += 1
        return result

    def _unary_op(op):
        def _(self):
            return DenseVector(getattr(self._values, op)())

        return _

    def _binary_op(op):
        def _(self, other):
            if isinstance(other, DenseVector):
                other = other._values
            return DenseVector(getattr(self._values, op)(other))

        return _

    # arithmetic functions
    __add__ = _binary_op("__add__")  # type: ignore
    __sub__ = _binary_op("__sub__")  # type: ignore
    __mul__ = _binary_op("__mul__")  # type: ignore
    __truediv__ = _binary_op("__truediv__")  # type: ignore
    __mod__ = _binary_op("__mod__")  # type: ignore
    __radd__ = _binary_op("__radd__")  # type: ignore
    __rsub__ = _binary_op("__rsub__")  # type: ignore
    __rmul__ = _binary_op("__rmul__")  # type: ignore
    __rtruediv__ = _binary_op("__rtruediv__")  # type: ignore
    __rmod__ = _binary_op("__rmod__")  # type: ignore
    __neg__ = _unary_op("__neg__")  # type: ignore


class SparseVector(Vector):
    """
    A sparse vector, using either a dict, a list of (index, value) pairs, or two separate
    arrays of indices and values.

    Example:
    ::

        >>> Vectors.sparse(4, {1: 1.0, 3: 5.5})
        SparseVector(4, {1: 1.0, 3: 5.5})
        >>> Vectors.sparse(4, [(1, 1.0), (3, 5.5)])
        SparseVector(4, {1: 1.0, 3: 5.5})
        >>> Vectors.sparse(4, [1, 3], [1.0, 5.5])
        SparseVector(4, {1: 1.0, 3: 5.5})
    """

    def __init__(self, size: int, *args):
        self._size = size
        assert 1 <= len(args) <= 2, "The number of arguments must be 2 or 3"
        if len(args) == 1:
            # a dict, a list of (index, value) pairs
            pairs = args[0]
            if isinstance(pairs, dict):
                pairs = pairs.items()
            pairs = sorted(pairs)
            self._indices = np.array([p[0] for p in pairs], dtype=np.int32)
            self._values = np.array([p[1] for p in pairs], dtype=np.float64)
        else:
            assert len(args[0]) == len(args[1]), "The length of indices and values should be same"
            # two separate arrays of indices and values.
            self._indices = np.array(args[0], dtype=np.int32)
            self._values = np.array(args[1], dtype=np.float64)
            for i in range(len(self._indices) - 1):
                if self._indices[i] >= self._indices[i + 1]:
                    raise TypeError(
                        "Indices {0} and {1} are not strictly increasing".format(
                            self._indices[i], self._indices[i + 1]))

    def size(self) -> int:
        return self._size

    def get(self, i: int):
        idx = self._indices.searchsorted(i)
        if idx < len(self._indices) and self._indices[idx] == i:
            return self._values[idx]
        else:
            return 0.0

    def set(self, i: int, value: np.float64):
        idx = self._indices.searchsorted(i)
        if idx < len(self._indices) and self._indices[idx] == i:
            self._values[idx] = value
        elif value != 0:
            assert i < self._size
            cur_len = len(self._indices)
            indices = np.zeros(cur_len + 1, dtype=np.int32)
            values = np.zeros(cur_len + 1, dtype=np.float64)
            indices[0:idx] = self._indices[0:idx]
            values[0:idx] = self._values[0:idx]
            indices[idx] = i
            values[idx] = value
            indices[idx + 1:] = self._indices[idx:]
            values[idx + 1:] = self._values[idx]
            self._indices = indices
            self._values = values

    def to_array(self) -> np.ndarray:
        """
        Returns a copy of this SparseVector as a 1-dimensional NumPy array.
        """
        arr = np.zeros((self._size,), dtype=np.float64)
        arr[self._indices] = self._values
        return arr

    def dot(self, other: Union[Vector, np.ndarray, Sized]) -> np.ndarray:
        """
        Dot product of two Vectors.

        Examples:
        ::

            >>> sparse = SparseVector(4, [1, 3], [3.0, 4.0])
            >>> sparse.dot(sparse)
            25.0
            >>> sparse.dot(array.array('d', [1., 2., 3., 4.]))
            22.0
            >>> sparse2 = SparseVector(4, [2], [1.0])
            >>> sparse.dot(sparse2)
            0.0
            >>> sparse.dot(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))
            array([22., 22.])
        """
        if isinstance(other, np.ndarray):
            if other.ndim not in (2, 1):
                raise ValueError('Cannot call dot with %d-dimensional array' % other.ndim)
            assert len(self) == other.shape[0], "dimension mismatch"
            return np.dot(self._values, other[self._indices])

        assert len(self) == len(other), "dimension mismatch"

        if isinstance(other, DenseVector):
            return np.dot(other.to_array()[self._indices], self._values)
        elif isinstance(other, SparseVector):
            self_cmind = np.in1d(self._indices, other._indices, assume_unique=True)
            self_values = self._values[self_cmind]
            if self_values.size == 0:
                return np.float_(0.0)
            else:
                other_cmind = np.in1d(other._indices, self._indices, assume_unique=True)
                return np.dot(self_values, other._values[other_cmind])
        else:
            if isinstance(other, (array.array, np.ndarray, list, tuple, range)):
                return self.dot(DenseVector(other))
            raise ValueError('Cannot call with the type %s' % (type(other)))

    def squared_distance(self, other: Union[Vector, np.ndarray, Sized]) -> np.ndarray:
        """
        Squared distance of two Vectors.

        Examples:
        ::

            >>> sparse = SparseVector(4, [1, 3], [3.0, 4.0])
            >>> sparse.squared_distance(sparse)
            0.0
            >>> sparse.squared_distance(array.array('d', [1., 2., 3., 4.]))
            11.0
            >>> sparse.squared_distance(np.array([1., 2., 3., 4.]))
            11.0
            >>> sparse2 = SparseVector(4, [2], [1.0])
            >>> sparse.squared_distance(sparse2)
            26.0
            >>> sparse2.squared_distance(sparse)
            26.0
        """
        assert len(self) == len(other), "dimension mismatch"

        if isinstance(other, np.ndarray) or isinstance(other, DenseVector):
            if isinstance(other, np.ndarray) and other.ndim != 1:
                raise ValueError(
                    "Cannot call squared_distance with %d-dimensional array" % other.ndim)

            if isinstance(other, DenseVector):
                other = other.to_array()
            sparse_ind = np.zeros(other.size, dtype=bool)
            sparse_ind[self._indices] = True
            dist = other[sparse_ind] - self._values
            result = np.dot(dist, dist)

            other_ind = other[~sparse_ind]
            result += np.dot(other_ind, other_ind)
            return result
        elif isinstance(other, SparseVector):
            i = 0
            j = 0
            result = 0.0
            while i < len(self._indices) and j < len(other._indices):
                if self._indices[i] == other._indices[j]:
                    diff = self._values[i] - other._values[j]
                    result += diff * diff
                    i += 1
                    j += 1
                elif self._indices[i] < other._indices[j]:
                    result += self._values[i] * self._values[i]
                    i += 1
                else:
                    result += other._values[j] * other._values[j]
                    j += 1
            while i < len(self._indices):
                result += self._values[i] * self._values[i]
                i += 1
            while j < len(other._indices):
                result += other._values[j] * other._values[j]
                j += 1
            return np.float_(result)
        else:
            if isinstance(other, (array.array, np.ndarray, list, tuple, range)):
                return self.squared_distance(DenseVector(other))
            raise ValueError('Cannot call with the type %s' % (type(other)))

    def __len__(self):
        return self.size()

    def __eq__(self, other):
        if isinstance(other, SparseVector):
            return (other.size() == self.size()
                    and np.array_equal(other._indices, self._indices)
                    and np.array_equal(other._values, self._values))
        elif isinstance(other, DenseVector):
            if self.size != len(other):
                return False
            return Vector._equals(
                self._indices, self._values, list(range(len(other))), other.to_array())
        return False

    def __str__(self):
        inds = "[" + ",".join([str(i) for i in self._indices]) + "]"
        vals = "[" + ",".join([str(v) for v in self._values]) + "]"
        return "(" + ",".join((str(self.size()), inds, vals)) + ")"

    def __repr__(self):
        inds = self._indices
        vals = self._values
        entries = ", ".join(["{0}: {1}".format(inds[i], float(vals[i])) for i in range(len(inds))])
        return "SparseVector({0}, {{{1}}})".format(self._size, entries)


class Vectors(object):

    @staticmethod
    def dense(*elements) -> DenseVector:
        """
        Create a dense vector of 64-bit floats from a Python list or numbers.

        Examples:
        ::

            >>> Vectors.dense([1, 2, 3])
            DenseVector([1.0, 2.0, 3.0])
            >>> Vectors.dense(1.0, 2.0)
            DenseVector([1.0, 2.0])
        """
        if len(elements) == 1 and not isinstance(elements[0], (float, int)):
            # it's list, numpy.array or other iterable object.
            elements = elements[0]
        return DenseVector(elements)

    @staticmethod
    def sparse(size: int, *args):
        """
        Create a sparse vector, using either a dict, a list of (index, value) pairs, or two separate
        arrays of indices and values.

        Examples:
        ::

            >>> Vectors.sparse(4, {1: 1.0, 3: 5.5})
            SparseVector(4, {1: 1.0, 3: 5.5})
            >>> Vectors.sparse(4, [(1, 1.0), (3, 5.5)])
            SparseVector(4, {1: 1.0, 3: 5.5})
            >>> Vectors.sparse(4, [1, 3], [1.0, 5.5])
            SparseVector(4, {1: 1.0, 3: 5.5})

        :param size: The size of the vector.
        :param args: Non-zero entries, as a dictionary, list of tuples,
                     or two sorted lists containing indices and values.
        """
        return SparseVector(size, *args)


class Matrix(ABC):
    """
    A matrix of double values.
    """

    @abstractmethod
    def num_rows(self) -> int:
        pass

    @abstractmethod
    def num_cols(self) -> int:
        pass

    @abstractmethod
    def get(self, i: int, j: int) -> float:
        pass

    @abstractmethod
    def to_array(self) -> np.ndarray:
        """
        Convert the matrix into an numpy.ndarray
        """
        pass


class DenseMatrix(Matrix):
    """
    Column-major dense matrix. The entry values are stored in a single array of doubles with columns
    listed in sequence.
    """

    def __init__(self, num_rows: int, num_cols: int, values):
        assert len(values) == num_rows * num_cols
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._values = np.asarray(values, dtype=np.float64)

    def num_rows(self) -> int:
        return self._num_rows

    def num_cols(self) -> int:
        return self._num_cols

    def get(self, i: int, j: int) -> float:
        if i < 0 or i >= self._num_rows:
            raise IndexError("Row index %d is out of range [0, %d)" % (i, self._num_rows))
        if j >= self._num_cols or j < 0:
            raise IndexError("Column index %d is out of range [0, %d)" % (j, self._num_cols))
        return self._values[self._num_rows * j + i]

    def to_array(self) -> np.ndarray:
        """
        Return a numpy.ndarray

        Examples:
        ::

            >>> m = DenseMatrix(2, 2, range(4))
            >>> m.to_array()
            array([[ 0., 2.],
                   [ 1., 3.]])
        """
        return self._values.reshape((self._num_rows, self._num_cols), order="F")

    def __getitem__(self, indices):
        i, j = indices
        return self.get(i, j)

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if self._num_rows != other.num_rows() or self._num_cols != other.num_cols():
            return False

        self_values = np.ravel(self.to_array(), order="F")
        other_values = np.ravel(other.to_array(), order="F")
        return np.all(self_values == other_values)


def _double_to_long_bits(value: float) -> int:
    if np.isnan(value):
        value = float("nan")
    # pack double into 64 bits, then unpack as long int
    return struct.unpack("Q", struct.pack("d", value))[0]
