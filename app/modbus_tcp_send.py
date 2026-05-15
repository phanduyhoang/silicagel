import socket
import struct
import time


def build_mbap(transaction_id: int, unit_id: int, pdu: bytes) -> bytes:
    protocol_id = 0
    length = len(pdu) + 1
    return struct.pack(">HHHB", transaction_id, protocol_id, length, unit_id)


def send_modbus_tcp(ip: str, port: int, unit: int, pdu: bytes) -> bytes:
    tid = 1
    mbap = build_mbap(tid, unit, pdu)
    frame = mbap + pdu
    with socket.create_connection((ip, port), timeout=2) as sock:
        sock.sendall(frame)
        sock.settimeout(2)
        return sock.recv(256)


def write_single_register(ip: str, port: int, unit_id: int, register: int, value: int) -> bytes:
    pdu = struct.pack(">BHH", 0x06, int(register) & 0xFFFF, int(value) & 0xFFFF)
    return send_modbus_tcp(ip, port, unit_id, pdu)


def pulse_register(
    ip: str,
    port: int = 502,
    unit_id: int = 1,
    register: int = 0x01D6,
    on_value: int = 1,
    off_value: int = 0,
    hold_seconds: float = 0.5,
) -> None:
    write_single_register(ip, port, unit_id, register, on_value)
    time.sleep(max(0.0, float(hold_seconds)))
    write_single_register(ip, port, unit_id, register, off_value)

