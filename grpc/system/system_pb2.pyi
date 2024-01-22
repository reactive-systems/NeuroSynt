"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.enum_type_wrapper
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _System:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _SystemEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_System.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    SYSTEM_AIGER: _System.ValueType  # 0
    SYSTEM_MEALY: _System.ValueType  # 1

class System(_System, metaclass=_SystemEnumTypeWrapper):
    """All available system types
    Can easily be extended without breaking backwards compatibility
    """

SYSTEM_AIGER: System.ValueType  # 0
SYSTEM_MEALY: System.ValueType  # 1
global___System = System