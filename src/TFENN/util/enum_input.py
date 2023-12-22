# Copyright 2023 Kévin Garanger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from enum import Enum


class EnumInputClass(Enum):
    """
    This abstract class is used to create enum types that correspond to specific groups
    of classes. Classes in a group may or may not inherit from a common base class.
    For a given Enum inheriting from EnumInputClass and corresponding to a group of
    classes, the possible values of the enum correspond to the classes in the group.
    The 'obj_map' class method returns a dictionary that links enum values to their
    corresponding class and must be implemented in classes inheriting from
    EnumInputClass.
    The 'create' class method returns an instance of the class corresponding to the
    given enum value.
    """

    @classmethod
    @property
    @abc.abstractmethod
    def obj_map(cls) -> dict[Enum, object]:
        pass

    def create(
        self,
        *args,
        **kwargs,
    ) -> object:
        return self.__class__.obj_map[self](*args, **kwargs)


class EnumInputFun(Enum):
    """
    This abstract class is used to create enum types that correspond to specific groups
    of functions. For a given Enum inheriting from EnumInputFun and corresponding to a
    group of functions, the possible values of the enum correspond to the functions in
    the group.
    The 'fun_map' class method returns a dictionary that links enum values to their
    corresponding function and must be implemented in classes inheriting from
    EnumInputFun.
    The 'create' class method returns the function corresponding to the given enum
    value.
    """

    @classmethod
    @property
    @abc.abstractmethod
    def fun_map(cls) -> dict[Enum, callable]:
        pass

    def create(
        self,
        *args,
        **kwargs,
    ) -> callable:
        return self.fun_map[self]
