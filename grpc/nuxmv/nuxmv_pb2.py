# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ml2/grpc/nuxmv/nuxmv.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from ml2.grpc.ltl import ltl_mc_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2
from ml2.grpc.tools import tools_pb2 as ml2_dot_grpc_dot_tools_dot_tools__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1aml2/grpc/nuxmv/nuxmv.proto\x1a\x19ml2/grpc/ltl/ltl_mc.proto\x1a\x1aml2/grpc/tools/tools.proto2\xd8\x01\n\x05Nuxmv\x12(\n\x05Setup\x12\r.SetupRequest\x1a\x0e.SetupResponse\"\x00\x12=\n\x08Identify\x12\x16.IdentificationRequest\x1a\x17.IdentificationResponse\"\x00\x12-\n\nModelCheck\x12\r.LTLMCProblem\x1a\x0e.LTLMCSolution\"\x00\x12\x37\n\x10ModelCheckStream\x12\r.LTLMCProblem\x1a\x0e.LTLMCSolution\"\x00(\x01\x30\x01\x62\x06proto3')



_NUXMV = DESCRIPTOR.services_by_name['Nuxmv']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _NUXMV._serialized_start=86
  _NUXMV._serialized_end=302
# @@protoc_insertion_point(module_scope)
