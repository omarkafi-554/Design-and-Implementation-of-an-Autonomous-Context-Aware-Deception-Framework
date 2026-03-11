# Zeek Logs Reference

## conn.log — Network Connections
| Field | Type | Description |
|-------|------|-------------|
| ts | time | Timestamp |
| id.orig_h | addr | Source IP |
| id.orig_p | port | Source Port |
| id.resp_h | addr | Destination IP |
| id.resp_p | port | Destination Port |
| proto | enum | Protocol (tcp/udp) |
| duration | interval | Connection duration |
| orig_bytes | count | Bytes sent |
| resp_bytes | count | Bytes received |
| conn_state | string | Connection state |

## ssh.log — SSH Connections
| Field | Type | Description |
|-------|------|-------------|
| id.orig_h | addr | Attacker IP |
| auth_success | bool | Login success? |
| auth_attempts | count | Number of attempts |
| client | string | SSH client version |

## http.log — HTTP Requests
| Field | Type | Description |
|-------|------|-------------|
| id.orig_h | addr | Client IP |
| method | string | GET/POST/etc |
| host | string | Target hostname |
| uri | string | Requested path |
| status_code | count | Response code |

## dns.log — DNS Queries
| Field | Type | Description |
|-------|------|-------------|
| id.orig_h | addr | Client IP |
| query | string | Domain queried |
| qtype_name | string | Query type (A/AAAA) |
| answers | string | DNS response |
