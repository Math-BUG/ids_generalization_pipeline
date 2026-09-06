"""Shared schema constants and column-name aliases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    """Storage is independent of semantic meaning and model treatment."""

    storage_dtype: str
    semantic_type: str
    model_treatment: str
    meaning: str
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    integral: bool = False
    nullable: bool = True
    missing_tokens: tuple[str, ...] = ("", "-")
    log1p: bool = False


TON_IOT_SCHEMA_VERSION = "ton_iot_network/1.0.0"


def _quantity(meaning: str, unit: str, *, integral: bool = True) -> ColumnSpec:
    # float64 plus NaN is interoperable with both sklearn and the pandas/CuPy path.
    return ColumnSpec("float64", "count" if integral else "continuous", "numeric",
                      meaning, unit, minimum=0, integral=integral, log1p=True)


def _category(meaning: str, semantic_type: str = "nominal") -> ColumnSpec:
    return ColumnSpec("string", semantic_type, "categorical", meaning)


def _code(meaning: str) -> ColumnSpec:
    # Zero is retained: its meaning in the processed TON_IoT files is ambiguous.
    return ColumnSpec("string", "nominal_code", "categorical", meaning,
                      minimum=0, integral=True)


# Union of the headers of Network_dataset_1.csv ... Network_dataset_23.csv.
# Only partition 6 contains uid. Sources and unresolved encodings are documented
# in docs/ton_iot_schema.md. This is NOT the behavioral feature allowlist.
TON_IOT_SCHEMA = {
    "ts": ColumnSpec("float64", "timestamp", "numeric", "Connection start time",
                     "Unix seconds", minimum=0),
    "uid": _category("Connection identifier", "identifier"),
    "src_ip": _category("Source IP address", "identifier"),
    "src_port": ColumnSpec("Int64", "port", "categorical", "Source port",
                           minimum=0, maximum=65535, integral=True),
    "dst_ip": _category("Destination IP address", "identifier"),
    "dst_port": ColumnSpec("Int64", "port", "categorical", "Destination port",
                           minimum=0, maximum=65535, integral=True),
    "proto": _category("Transport protocol"),
    "service": _category("Application service"),
    "duration": _quantity("Connection duration", "seconds", integral=False),
    "src_bytes": _quantity("Source payload bytes", "bytes"),
    "dst_bytes": _quantity("Destination payload bytes", "bytes"),
    "conn_state": _category("Connection state"),
    "missed_bytes": _quantity("Bytes missed in content gaps", "bytes"),
    "src_pkts": _quantity("Source packets", "packets"),
    "src_ip_bytes": _quantity("Source IP-level bytes", "bytes"),
    "dst_pkts": _quantity("Destination packets", "packets"),
    "dst_ip_bytes": _quantity("Destination IP-level bytes", "bytes"),
    "dns_query": _category("DNS query name", "text"),
    "dns_qclass": _code("DNS query class code"),
    "dns_qtype": _code("DNS query type code"),
    "dns_rcode": _code("DNS response code"),
    "dns_AA": _category("DNS authoritative-answer flag", "boolean"),
    "dns_RD": _category("DNS recursion-desired flag", "boolean"),
    "dns_RA": _category("DNS recursion-available flag", "boolean"),
    "dns_rejected": _category("DNS query rejected flag", "boolean"),
    "ssl_version": _category("SSL/TLS version"),
    "ssl_cipher": _category("SSL/TLS cipher suite"),
    "ssl_resumed": _category("SSL/TLS session resumed flag", "boolean"),
    "ssl_established": _category("SSL/TLS session established flag", "boolean"),
    "ssl_subject": _category("Certificate subject", "text"),
    "ssl_issuer": _category("Certificate issuer", "text"),
    "http_trans_depth": _quantity("HTTP transaction pipeline depth", "transactions"),
    "http_method": _category("HTTP request method"),
    "http_uri": _category("HTTP request URI", "text"),
    "http_referrer": _category("HTTP referrer", "text"),
    "http_version": _category("HTTP version"),
    "http_request_body_len": _quantity("HTTP request body length", "bytes"),
    "http_response_body_len": _quantity("HTTP response body length", "bytes"),
    "http_status_code": _code("HTTP response status code"),
    "http_user_agent": _category("HTTP user agent", "text"),
    "http_orig_mime_types": _category("Originator MIME types (opaque serialized value)", "text"),
    "http_resp_mime_types": _category("Responder MIME types (opaque serialized value)", "text"),
    "weird_name": _category("Zeek unusual-activity name"),
    "weird_addl": _category("Additional unusual-activity information", "text"),
    "weird_notice": _category("Unusual activity also raised a notice", "boolean"),
    "label": ColumnSpec("Int64", "binary_target", "target", "0 normal, 1 attack",
                        minimum=0, maximum=1, integral=True, nullable=False),
    "type": ColumnSpec("string", "multiclass_target", "target", "Attack type or normal",
                       nullable=False),
}

TON_IOT_LOG1P_COLUMNS = tuple(name for name, spec in TON_IOT_SCHEMA.items() if spec.log1p)

# Existing extract_causal outputs, kept separate from the 47 raw dataset fields.
# -1 is the existing extractor's missing-value code, retained for compatibility.
TON_IOT_DERIVED_SCHEMA = {
    "ts_hour": ColumnSpec("Int64", "calendar_component", "numeric", "Derived hour; -1 unavailable",
                          minimum=-1, maximum=23, integral=True),
    "ts_day_of_week": ColumnSpec("Int64", "calendar_component", "numeric", "Derived weekday; -1 unavailable",
                                 minimum=-1, maximum=6, integral=True),
}

# Explicit compatibility list for the bundled synthetic data; no name matching.
DEFAULT_LOG1P_COLUMNS = TON_IOT_LOG1P_COLUMNS

TARGET_COLUMNS = {
    "label",
    "type",
    "target",
    "attack_cat",
    "class",
    "cluster_id",
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

SERVICE_COLUMNS = {"service", "svc", "application", "app", "proto_service"}

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
    "hour",
    "day_of_week",
    "median",
    "iat",
    "jitter",
    "loss",
    "load",
    "tcp",
    "udp",
    "icmp",
)
