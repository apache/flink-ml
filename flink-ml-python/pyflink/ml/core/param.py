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

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Dict, Any, Optional, Tuple, Union

import jsonpickle

T = TypeVar('T')
V = TypeVar('V')


class WithParams(Generic[T], ABC):
    """
    Interface for classes that take parameters. It provides APIs to set and get parameters.
    """

    def set(self, param: 'Param[V]', value: V) -> 'WithParams[T]':
        """
        Sets the value of the parameter.

        :param param: The parameter.
        :param value: The parameter value.
        :return: The WithParams instance itself.
        """
        if not self._is_compatible_type(param, value):
            raise TypeError(
                f"Parameter {param.name}'s type {param.type} is incompatible with the type of "
                f"{type(value)}")

        if not param.validator.validate(value):
            if value is None:
                raise ValueError(f'Parameter {param.name}\'s value should not be None.')
            else:
                raise ValueError(f'Parameter {param.name} is given an invalid value {value}.')

        self.get_param_map()[param] = value
        return self

    def get_param(self, name: str) -> Optional['Param[V]']:
        """
        Gets the parameter by its name.

        :param name: The parameter name.
        :return: The parameter.
        """
        for item in self.get_param_map():
            if item.name == name:
                return item
        return None

    def get(self, param: 'Param[V]') -> V:
        """
        Gets the value of the parameter.

        :param param: The parameter.
        :return: The parameter value.
        """
        value = self.get_param_map().get(param)
        if value is None and not param.validator.validate(None):
            raise ValueError(f'Parameter {param.name}\'s value should not be None')
        return value

    @abstractmethod
    def get_param_map(self) -> Dict['Param[Any]', Any]:
        """
        Returns a map which maps parameter definition to parameter value.
        """
        pass

    @staticmethod
    def _is_compatible_type(param: 'Param[V]', value: V) -> bool:
        if value is not None and param.type != type(value):
            return False
        if isinstance(value, list):
            for item in value:
                if param.type_name != f'list[{type(item).__name__}]':
                    return False
            return True
        return True


class ParamValidator(Generic[T], ABC):
    """
    An interface to validate that a parameter value is valid.
    """

    @abstractmethod
    def validate(self, value: T) -> bool:
        """
        Validates whether the parameter value is valid.

        :param value: The parameter value.
        :return: A boolean indicating whether the parameter value is valid.
        """
        pass


class ParamValidators(object):
    """
    Factory methods for common validation functions on numerical values.
    """

    @staticmethod
    def always_true() -> ParamValidator[T]:
        class AlwaysTrue(ParamValidator[T]):
            """
            Always return true.
            """

            def validate(self, value: T) -> bool:
                return True

        return AlwaysTrue()

    @staticmethod
    def gt(lower_bound: float) -> ParamValidator[T]:
        class GT(ParamValidator[T]):
            """
            Checks if the parameter value is greater than lower_bound.
            """

            def validate(self, value: T) -> bool:
                return value is not None and value > lower_bound  # type: ignore

        return GT()

    @staticmethod
    def gt_eq(lower_bound: float) -> ParamValidator[T]:
        class GtEq(ParamValidator[T]):
            """
            Checks if the parameter value is greater than or equal to lower_bound.
            """

            def validate(self, value: T) -> bool:
                return value is not None and value >= lower_bound  # type: ignore

        return GtEq()

    @staticmethod
    def lt(upper_bound: float) -> ParamValidator[T]:
        class LT(ParamValidator[T]):
            """
            Checks if the parameter value is less than upper_bound.
            """

            def validate(self, value: T) -> bool:
                return value is not None and value < upper_bound  # type: ignore

        return LT()

    @staticmethod
    def lt_eq(upper_bound: float) -> ParamValidator[T]:
        """
         Checks if the parameter value is less than or equal to upper_bound.
         """

        class LtEq(ParamValidator[T]):
            def validate(self, value: T) -> bool:
                return value is not None and value <= upper_bound  # type: ignore

        return LtEq()

    @staticmethod
    def in_range(lower_bound: float, upper_bound: float, lower_inclusive: bool = True,
                 upper_inclusive: bool = True) -> ParamValidator[T]:
        """
        Checks if the parameter value is in the range from lower_bound to upper_bound.
        """

        class InRange(ParamValidator[T]):
            def validate(self, value: T) -> bool:
                return (value is not None
                        and lower_bound <= value <= upper_bound  # type: ignore
                        and (lower_inclusive or value != lower_bound)
                        and (upper_inclusive or value != upper_bound))

        return InRange()

    @staticmethod
    def in_array(allowed: Union[Tuple[T], List[T]]) -> ParamValidator[T]:
        """
        Checks if the parameter value is in the array of allowed values.
        """

        class InArray(ParamValidator[T]):
            def validate(self, value: T) -> bool:
                return value is not None and value in allowed

        return InArray()

    @staticmethod
    def not_null() -> ParamValidator[T]:
        """
        Checks if the parameter value is not None.
        """

        class NotNull(ParamValidator[T]):
            def validate(self, value: T) -> bool:
                return value is not None

        return NotNull()


class Param(Generic[T]):
    """
    Definition of a parameter, including name, description, default value and the validator.
    """

    def __init__(self, name: str, type_type: type, type_name: str, description: str,
                 default_value: T, validator: ParamValidator[T]):
        self.name = name
        self.type = type_type
        self.type_name = type_name
        self.description = description
        self.default_value = default_value
        self.validator = validator
        if default_value is not None and not validator.validate(default_value):
            raise ValueError(f"Parameter {name} is given an invalid value {default_value}")

    @staticmethod
    def json_encode(value: T) -> str:
        """
        Encodes the given object into a json-formatted string.

        :param value: An object of class type T.
        :return: A json-formatted string.
        """
        return str(jsonpickle.encode(value, keys=True))

    @staticmethod
    def json_decode(json: str) -> T:
        """
        Decodes the given string into an object of class type T.

        :param json: A json-formatted string.
        :return: An object of class type T.
        """
        return jsonpickle.decode(json, keys=True)

    def __eq__(self, other):
        return isinstance(other, Param) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


class BooleanParam(Param[bool]):
    """
    Class for the boolean parameter.
    """

    def __init__(self, name: str, description: str, default_value: Optional[bool],
                 validator: ParamValidator[bool] = ParamValidators.always_true()):
        super(BooleanParam, self).__init__(name, bool, "bool", description, default_value,
                                           validator)


class IntParam(Param[int]):
    """
    Class for the int parameter.
    """

    def __init__(self, name: str, description: str, default_value: Optional[int],
                 validator: ParamValidator[int] = ParamValidators.always_true()):
        super(IntParam, self).__init__(name, int, "int", description, default_value, validator)


class FloatParam(Param[float]):
    """
    Class for the float parameter.
    """

    def __init__(self, name: str, description: str, default_value: Optional[float],
                 validator: ParamValidator[float] = ParamValidators.always_true()):
        super(FloatParam, self).__init__(name, float, "float", description, default_value,
                                         validator)


class StringParam(Param[str]):
    """
    Class for the string parameter.
    """

    def __init__(self, name: str, description: str, default_value: Optional[str],
                 validator: ParamValidator[str] = ParamValidators.always_true()):
        super(StringParam, self).__init__(name, str, "str", description, default_value, validator)


class IntArrayParam(Param[List[int]]):
    """
    Class for the string parameter.
    """

    def __init__(self, name: str, description: str, default_value: Optional[List[int]],
                 validator: ParamValidator[List[int]] = ParamValidators.always_true()):
        super(IntArrayParam, self).__init__(name, list, "list[int]", description, default_value,
                                            validator)


class FloatArrayParam(Param[List[float]]):
    """
    Class for the string parameter.
    """

    def __init__(self, name: str, description: str, default_value: Optional[List[float]],
                 validator: ParamValidator[List[float]] = ParamValidators.always_true()):
        super(FloatArrayParam, self).__init__(name, list, "list[float]", description, default_value,
                                              validator)


class StringArrayParam(Param[List[str]]):
    """
    Class for the string array parameter.
    """

    def __init__(self, name: str, description: str, default_value: Optional[List[str]],
                 validator: ParamValidator[List[str]] = ParamValidators.always_true()):
        super(StringArrayParam, self).__init__(name, list, "list[str]", description, default_value,
                                               validator)
