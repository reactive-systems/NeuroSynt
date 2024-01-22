# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ml2/grpc/spot/spot.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from ml2.grpc.aiger import aiger_pb2 as ml2_dot_grpc_dot_aiger_dot_aiger__pb2
from ml2.grpc.ltl import ltl_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__pb2
from ml2.grpc.ltl import ltl_equiv_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__equiv__pb2
from ml2.grpc.ltl import ltl_mc_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__mc__pb2
from ml2.grpc.ltl import ltl_sat_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__sat__pb2
from ml2.grpc.ltl import ltl_syn_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__syn__pb2
from ml2.grpc.ltl import ltl_trace_mc_pb2 as ml2_dot_grpc_dot_ltl_dot_ltl__trace__mc__pb2
from ml2.grpc.mealy import mealy_pb2 as ml2_dot_grpc_dot_mealy_dot_mealy__pb2
from ml2.grpc.system import system_pb2 as ml2_dot_grpc_dot_system_dot_system__pb2
from ml2.grpc.tools import tools_pb2 as ml2_dot_grpc_dot_tools_dot_tools__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18ml2/grpc/spot/spot.proto\x1a\x1aml2/grpc/aiger/aiger.proto\x1a\x16ml2/grpc/ltl/ltl.proto\x1a\x1cml2/grpc/ltl/ltl_equiv.proto\x1a\x19ml2/grpc/ltl/ltl_mc.proto\x1a\x1aml2/grpc/ltl/ltl_sat.proto\x1a\x1aml2/grpc/ltl/ltl_syn.proto\x1a\x1fml2/grpc/ltl/ltl_trace_mc.proto\x1a\x1aml2/grpc/mealy/mealy.proto\x1a\x1cml2/grpc/system/system.proto\x1a\x1aml2/grpc/tools/tools.proto\"\xe5\x01\n\x0bRandLTLArgs\x12\x14\n\x0cnum_formulas\x18\x01 \x01(\x05\x12\x0f\n\x07num_aps\x18\x02 \x01(\x05\x12\x0b\n\x03\x61ps\x18\x03 \x03(\t\x12\x12\n\nallow_dups\x18\x04 \x01(\x08\x12\x0e\n\x06output\x18\x05 \x01(\t\x12\x0c\n\x04seed\x18\x06 \x01(\x05\x12\x10\n\x08simplify\x18\x07 \x01(\x05\x12\x11\n\ttree_size\x18\x08 \x01(\x05\x12\x1a\n\x12\x62oolean_priorities\x18\t \x01(\t\x12\x16\n\x0eltl_priorities\x18\n \x01(\t\x12\x17\n\x0fsere_priorities\x18\x0b \x01(\t2\xe0\x04\n\x04Spot\x12(\n\x05Setup\x12\r.SetupRequest\x1a\x0e.SetupResponse\"\x00\x12=\n\x08Identify\x12\x16.IdentificationRequest\x1a\x17.IdentificationResponse\"\x00\x12-\n\nModelCheck\x12\r.LTLMCProblem\x1a\x0e.LTLMCSolution\"\x00\x12\x37\n\x10ModelCheckStream\x12\r.LTLMCProblem\x1a\x0e.LTLMCSolution\"\x00(\x01\x30\x01\x12/\n\nSynthesize\x12\x0e.LTLSynProblem\x1a\x0f.LTLSynSolution\"\x00\x12\x33\n\nCheckEquiv\x12\x10.LTLEquivProblem\x1a\x11.LTLEquivSolution\"\x00\x12-\n\x08\x43heckSat\x12\x0e.LTLSatProblem\x1a\x0f.LTLSatSolution\"\x00\x12\x34\n\x07MCTrace\x12\x12.LTLTraceMCProblem\x1a\x13.LTLTraceMCSolution\"\x00\x12(\n\x07RandLTL\x12\x0c.RandLTLArgs\x1a\x0b.LTLFormula\"\x00\x30\x01\x12+\n\taag2mealy\x12\r.AigerCircuit\x1a\r.MealyMachine\"\x00\x12+\n\tmealy2aag\x12\r.MealyMachine\x1a\r.AigerCircuit\"\x00\x12\x38\n\x12\x65xtractTransitions\x12\r.MealyMachine\x1a\x11.MealyTransitions\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ml2.grpc.spot.spot_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_RANDLTLARGS']._serialized_start=313
  _globals['_RANDLTLARGS']._serialized_end=542
  _globals['_SPOT']._serialized_start=545
  _globals['_SPOT']._serialized_end=1153
# @@protoc_insertion_point(module_scope)