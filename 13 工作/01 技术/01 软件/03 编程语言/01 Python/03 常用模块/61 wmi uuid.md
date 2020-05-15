# wmi uuid


```py
# 获得本机MAC地址：

import uuid

def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e + 2] for e in range(0, 11, 2)])
print(get_mac_address())


# 获取IP的方法：使用socket
import socket
hostname = socket.getfqdn(socket.gethostname())  # 获取本机电脑名
hostaddr = socket.gethostbyname(hostname)  # 获取本机ip 但是注意这里获取的IP是内网IP
print(hostname)
print(hostaddr)


# 在linux下可用
# import socket
# import fcntl
# import struct
#
#
# def get_ip_address(ifname):
#     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     return socket.inet_ntoa(fcntl.ioctl(
#         s.fileno(),
#         0x8915,  # SIOCGIFADDR
#         struct.pack('256s', ifname[:15])
#     )[20:24])
#
# get_ip_address('lo')
# get_ip_address('eth0')

```




wmi 好像没法用：

- [如何获取硬盘序列号使用 Python](https://codeday.me/bug/20171114/94418.html)
- [Python WMI模块的使用实例](https://blog.csdn.net/zmj_88888888/article/details/8700950)
- [Python在 windows 下获取 cpu、硬盘、bios、主板序列号](https://blog.csdn.net/xtx1990/article/details/7288903)
