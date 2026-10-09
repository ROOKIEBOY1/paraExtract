"""Hard network guard used by offline validation subprocesses."""


def block_network() -> None:
    import socket

    def denied(*args, **kwargs):
        raise RuntimeError("Network access blocked by --offline")

    class BlockedSocket(socket.socket):
        def connect(self, *args, **kwargs):
            denied()

        def connect_ex(self, *args, **kwargs):
            denied()

    socket.create_connection = denied
    socket.getaddrinfo = denied
    socket.socket = BlockedSocket
