# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: waymo_open_dataset/protos/segmentation.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='waymo_open_dataset/protos/segmentation.proto',
  package='waymo.open_dataset',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n,waymo_open_dataset/protos/segmentation.proto\x12\x12waymo.open_dataset\"\xd6\x03\n\x0cSegmentation\"\xc5\x03\n\x04Type\x12\x12\n\x0eTYPE_UNDEFINED\x10\x00\x12\x0c\n\x08TYPE_CAR\x10\x01\x12\x0e\n\nTYPE_TRUCK\x10\x02\x12\x0c\n\x08TYPE_BUS\x10\x03\x12\x16\n\x12TYPE_OTHER_VEHICLE\x10\x04\x12\x15\n\x11TYPE_MOTORCYCLIST\x10\x05\x12\x12\n\x0eTYPE_BICYCLIST\x10\x06\x12\x13\n\x0fTYPE_PEDESTRIAN\x10\x07\x12\r\n\tTYPE_SIGN\x10\x08\x12\x16\n\x12TYPE_TRAFFIC_LIGHT\x10\t\x12\r\n\tTYPE_POLE\x10\n\x12\x1a\n\x16TYPE_CONSTRUCTION_CONE\x10\x0b\x12\x10\n\x0cTYPE_BICYCLE\x10\x0c\x12\x13\n\x0fTYPE_MOTORCYCLE\x10\r\x12\x11\n\rTYPE_BUILDING\x10\x0e\x12\x13\n\x0fTYPE_VEGETATION\x10\x0f\x12\x13\n\x0fTYPE_TREE_TRUNK\x10\x10\x12\r\n\tTYPE_CURB\x10\x11\x12\r\n\tTYPE_ROAD\x10\x12\x12\x14\n\x10TYPE_LANE_MARKER\x10\x13\x12\x15\n\x11TYPE_OTHER_GROUND\x10\x14\x12\x11\n\rTYPE_WALKABLE\x10\x15\x12\x11\n\rTYPE_SIDEWALK\x10\x16')
)



_SEGMENTATION_TYPE = _descriptor.EnumDescriptor(
  name='Type',
  full_name='waymo.open_dataset.Segmentation.Type',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TYPE_UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_CAR', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_TRUCK', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_BUS', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_OTHER_VEHICLE', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_MOTORCYCLIST', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_BICYCLIST', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_PEDESTRIAN', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_SIGN', index=8, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_TRAFFIC_LIGHT', index=9, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_POLE', index=10, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_CONSTRUCTION_CONE', index=11, number=11,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_BICYCLE', index=12, number=12,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_MOTORCYCLE', index=13, number=13,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_BUILDING', index=14, number=14,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_VEGETATION', index=15, number=15,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_TREE_TRUNK', index=16, number=16,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_CURB', index=17, number=17,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_ROAD', index=18, number=18,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_LANE_MARKER', index=19, number=19,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_OTHER_GROUND', index=20, number=20,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_WALKABLE', index=21, number=21,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TYPE_SIDEWALK', index=22, number=22,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=86,
  serialized_end=539,
)
_sym_db.RegisterEnumDescriptor(_SEGMENTATION_TYPE)


_SEGMENTATION = _descriptor.Descriptor(
  name='Segmentation',
  full_name='waymo.open_dataset.Segmentation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SEGMENTATION_TYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=69,
  serialized_end=539,
)

_SEGMENTATION_TYPE.containing_type = _SEGMENTATION
DESCRIPTOR.message_types_by_name['Segmentation'] = _SEGMENTATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Segmentation = _reflection.GeneratedProtocolMessageType('Segmentation', (_message.Message,), {
  'DESCRIPTOR' : _SEGMENTATION,
  '__module__' : 'waymo_open_dataset.protos.segmentation_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.Segmentation)
  })
_sym_db.RegisterMessage(Segmentation)


# @@protoc_insertion_point(module_scope)