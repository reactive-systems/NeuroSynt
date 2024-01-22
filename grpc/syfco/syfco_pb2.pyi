"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.duration_pb2
import google.protobuf.internal.containers
import google.protobuf.message
import ml2.grpc.ltl.ltl_pb2
import sys
import typing

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class TLSFFileString(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TLSF_FIELD_NUMBER: builtins.int
    tlsf: builtins.str
    def __init__(
        self,
        *,
        tlsf: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["tlsf", b"tlsf"]) -> None: ...

global___TLSFFileString = TLSFFileString

@typing_extensions.final
class ConvertTLSFToSpecRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class ParametersEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    PARAMETERS_FIELD_NUMBER: builtins.int
    TLSF_FIELD_NUMBER: builtins.int
    @property
    def parameters(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]:
        """Defines run- and tool-specific parameters. As Map (Dict in Python).
        Typical examples are threads, timeouts etc. Can be empty.
        """
    @property
    def tlsf(self) -> global___TLSFFileString:
        """A string, read from a TLSF file"""
    def __init__(
        self,
        *,
        parameters: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
        tlsf: global___TLSFFileString | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["tlsf", b"tlsf"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["parameters", b"parameters", "tlsf", b"tlsf"]) -> None: ...

global___ConvertTLSFToSpecRequest = ConvertTLSFToSpecRequest

@typing_extensions.final
class ConvertTLSFToSpecResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SPECIFICATION_FIELD_NUMBER: builtins.int
    ERROR_FIELD_NUMBER: builtins.int
    TOOL_FIELD_NUMBER: builtins.int
    TIME_FIELD_NUMBER: builtins.int
    @property
    def specification(self) -> ml2.grpc.ltl.ltl_pb2.DecompLTLSpecification:
        """A string, read from a TLSF file"""
    error: builtins.str
    """Here additional information should be supplied if something went wrong"""
    tool: builtins.str
    """Tool that created the response"""
    @property
    def time(self) -> google.protobuf.duration_pb2.Duration:
        """How long the tool took to create the result."""
    def __init__(
        self,
        *,
        specification: ml2.grpc.ltl.ltl_pb2.DecompLTLSpecification | None = ...,
        error: builtins.str = ...,
        tool: builtins.str = ...,
        time: google.protobuf.duration_pb2.Duration | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["_specification", b"_specification", "_time", b"_time", "specification", b"specification", "time", b"time"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["_specification", b"_specification", "_time", b"_time", "error", b"error", "specification", b"specification", "time", b"time", "tool", b"tool"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_specification", b"_specification"]) -> typing_extensions.Literal["specification"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["_time", b"_time"]) -> typing_extensions.Literal["time"] | None: ...

global___ConvertTLSFToSpecResponse = ConvertTLSFToSpecResponse
