import subprocess
import sys


def test_offline_guard_blocks_socket_connections():
    code = "from pp_uie.offline import block_network; block_network(); import socket; socket.create_connection(('example.com', 443))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Network access blocked by --offline" in result.stderr


def test_offline_guard_blocks_dns_resolution_before_remote_lookup():
    code = "from pp_uie.offline import block_network; block_network(); import socket; socket.getaddrinfo('example.com', 443)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Network access blocked by --offline" in result.stderr
