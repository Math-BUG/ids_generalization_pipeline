"""Shared schema constants and column-name aliases."""

from __future__ import annotations

TARGET_COLUMNS = {
    "label",
    "type",
    "target",
    "attack_cat",
    "class",
}

RAW_IDENTITY_COLUMNS = {
    "src_ip",
    "dst_ip",
    "srcip",
    "dstip",
    "ip_src",
    "ip_dst",
    "src_port",
    "dst_port",
    "sport",
    "dport",
    "source_port",
    "destination_port",
    "uid",
    "flow_id",
    "id",
}

DATASET_ID_COLUMNS = {"dataset_id", "source_dataset"}

PROHIBITED_FEATURE_COLUMNS = TARGET_COLUMNS | RAW_IDENTITY_COLUMNS | DATASET_ID_COLUMNS

IP_COLUMNS = {"src_ip", "dst_ip", "srcip", "dstip", "ip_src", "ip_dst"}

PORT_COLUMNS = {
    "src_port",
    "dst_port",
    "sport",
    "dport",
    "source_port",
    "destination_port",
}

STABLE_IDENTIFIER_HINTS = (
    "uid",
    "flow_id",
    "session",
    "conn_id",
    "connection_id",
    "device_id",
    "host_id",
    "mac",
    "serial",
    "uuid",
    "user_id",
)

BEHAVIORAL_ALLOWED_HINTS = (
    "duration",
    "dur",
    "byte",
    "bytes",
    "pkt",
    "pkts",
    "packet",
    "packets",
    "flag",
    "state",
    "ttl",
    "window",
    "rate",
    "ratio",
    "count",
    "mean",
    "std",
    "min",
    "max",
    "sum",
    "avg",
    "median",
    "iat",
    "jitter",
    "loss",
    "load",
    "tcp",
    "udp",
    "icmp",
)
