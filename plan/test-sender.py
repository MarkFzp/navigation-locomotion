import socket, sys

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "localhost"
port = 8888
sock.sendto(str(sys.argv[1]).encode(), (ip, port))

