# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ml2/grpc/syfco/syfco.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import duration_pb2 as google_dot_protobuf_dot_duration__pb2
from ml2.grpc.ltl import ltl_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__pb2
from ml2.grpc.tools import tools_pb2 as ml2_dot_grpc_dot_tools_dot_tools__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1aml2/grpc/syfco/syfco.proto\x1a\x1egoogle/protobuf/duration.proto\x1a\x16ml2/grpc/ltl/ltl.proto\x1a\x1aml2/grpc/tools/tools.proto\"\x1e\n\x0eTLSFFileString\x12\x0c\n\x04tlsf\x18\x01 \x01(\t\"\xab\x01\n\x18\x43onvertTLSFToSpecRequest\x12=\n\nparameters\x18\x01 \x03(\x0b\x32).ConvertTLSFToSpecRequest.ParametersEntry\x12\x1d\n\x04tlsf\x18\x02 \x01(\x0b\x32\x0f.TLSFFileString\x1a\x31\n\x0fParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xb6\x01\n\x19\x43onvertTLSFToSpecResponse\x12\x33\n\rspecification\x18\x01 \x01(\x0b\x32\x17.DecompLTLSpecificationH\x00\x88\x01\x01\x12\r\n\x05\x65rror\x18\x02 \x01(\t\x12\x0c\n\x04tool\x18\x03 \x01(\t\x12,\n\x04time\x18\x04 \x01(\x0b\x32\x19.google.protobuf.DurationH\x01\x88\x01\x01\x42\x10\n\x0e_specificationB\x07\n\x05_time2\xbe\x01\n\x05Syfco\x12(\n\x05Setup\x12\r.SetupRequest\x1a\x0e.SetupResponse\"\x00\x12=\n\x08Identify\x12\x16.IdentificationRequest\x1a\x17.IdentificationResponse\"\x00\x12L\n\x11\x43onvertTLSFToSpec\x12\x19.ConvertTLSFToSpecRequest\x1a\x1a.ConvertTLSFToSpecResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ml2.grpc.syfco.syfco_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_CONVERTTLSFTOSPECREQUEST_PARAMETERSENTRY']._options = None
  _globals['_CONVERTTLSFTOSPECREQUEST_PARAMETERSENTRY']._serialized_options = b'8\001'
  _globals['_TLSFFILESTRING']._serialized_start=114
  _globals['_TLSFFILESTRING']._serialized_end=144
  _globals['_CONVERTTLSFTOSPECREQUEST']._serialized_start=147
  _globals['_CONVERTTLSFTOSPECREQUEST']._serialized_end=318
  _globals['_CONVERTTLSFTOSPECREQUEST_PARAMETERSENTRY']._serialized_start=269
  _globals['_CONVERTTLSFTOSPECREQUEST_PARAMETERSENTRY']._serialized_end=318
  _globals['_CONVERTTLSFTOSPECRESPONSE']._serialized_start=321
  _globals['_CONVERTTLSFTOSPECRESPONSE']._serialized_end=503
  _globals['_SYFCO']._serialized_start=506
  _globals['_SYFCO']._serialized_end=696
# @@protoc_insertion_point(module_scope)
