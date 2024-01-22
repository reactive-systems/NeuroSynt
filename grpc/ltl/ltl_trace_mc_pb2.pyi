"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class LTLTraceMCProblem(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FORMULA_FIELD_NUMBER: builtins.int
    TRACE_FIELD_NUMBER: builtins.int
    TIMEOUT_FIELD_NUMBER: builtins.int
    formula: builtins.str
    trace: builtins.str
    timeout: builtins.str
    def __init__(
        self,
        *,
        formula: builtins.str = ...,
        trace: builtins.str = ...,
        timeout: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["formula", b"formula", "timeout", b"timeout", "trace", b"trace"]) -> None: ...

global___LTLTraceMCProblem = LTLTraceMCProblem

@typing_extensions.final
class LTLTraceMCSolution(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STATUS_FIELD_NUMBER: builtins.int
    status: builtins.str
    def __init__(
        self,
        *,
        status: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["status", b"status"]) -> None: ...

global___LTLTraceMCSolution = LTLTraceMCSolution
