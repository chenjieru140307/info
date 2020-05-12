"socket" --- 底层网络接口
*************************

**源代码:** Lib/socket.py

======================================================================

这个模块提供了访问 BSD *套接字* 的接口。在所有现代 Unix 系统、Windows
、macOS 和其他一些平台上可用。

注解:

  一些行为可能因平台不同而异，因为调用的是操作系统的套接字API。

这个Python接口是用Python的面向对象风格对Unix系统调用和套接字库接口的直
译：函数 "socket()" 返回一个 *套接字对象* ，其方法是对各种套接字系统调
用的实现。形参类型一般与C接口相比更高级：例如在Python文件 "read()" 和
"write()" 操作中，接收操作的缓冲区分配是自动的，发送操作的缓冲区长度是
隐式的。

参见:

  模块 "socketserver"
     用于简化网络服务端编写的类。

  模块 "ssl"
     套接字对象的TLS/SSL封装。


套接字协议族
============

根据系统以及构建选项，此模块提供了各种套接字协议簇。

特定的套接字对象需要的地址格式将根据此套接字对象被创建时指定的地址族被
自动选择。套接字地址表示如下：

* 一个绑定在文件系统节点上的 "AF_UNIX" 套接字的地址表示为一个字符串，
  使用文件系统字符编码和 "'surrogateescape'" 错误回调方法（see **PEP
  383**）。一个地址在 Linux 的抽象命名空间被返回为带有初始的 null 字节
  的 *字节类对象* ；注意在这个命名空间种的套接字可能与普通文件系统套接
  字通信，所以打算运行在 Linux 上的程序可能需要解决两种地址类型。当传
  递为参数时，一个字符串或字节类对象可以用于任一类型的地址。

     在 3.3 版更改: 之前，"AF_UNIX" 套接字路径被假设使用 UTF-8 编码。

     在 3.5 版更改: 现在支持可写的 *字节类对象*。

* 一对 "(host, port)" 被用于 "AF_INET" 地址族，*host* 是一个表示为互联
  网域名表示法之内的主机名或者一个 IPv4 地址的字符串，例如
  "'daring.cwi.nl'" 或 "'100.50.200.5'"，*port* 是一个整数。

  * 对于 IPv4 地址，有两种可接受的特殊形式被用来代替一个主机地址：
    "''" 代表 "INADDR_ANY"，用来绑定到所有接口；字符串 "'<broadcast>'"
    代表 "INADDR_BROADCAST"。此行为不兼容 IPv6，因此，如果你的 Python
    程序打算支持 IPv6，则可能需要避开这些。

* 对于 "AF_INET6" 地址族，使用一个四元组 "(host, port, flowinfo,
  scopeid)"，其中 *flowinfo* 和 *scopeid* 代表了 C 库 "struct
  sockaddr_in6" 中的 "sin6_flowinfo" 和 "sin6_scope_id" 成员。对于
  "socket" 模块中的方法， *flowinfo* 和 *scopeid* 可以被省略，只为了向
  后兼容。注意，省略 *scopeid* 可能会导致操作带有领域 (Scope) 的 IPv6
  地址时出错。

  在 3.7 版更改: 对于多播地址（其 *scopeid* 起作用），*地址* 中可以不
  包含 "%scope" （或 "zone id" ）部分，这部分是多余的，可以放心省略（
  推荐）。

* "AF_NETLINK" 套接字由一对 "(pid, groups)" 表示。

* 指定 "AF_TIPC" 地址族可以使用仅 Linux 支持的 TIPC 协议。TIPC 是一种
  开放的、非基于 IP 的网络协议，旨在用于集群计算环境。其地址用元组表示
  ，其中的字段取决于地址类型。一般元组形式为 "(addr_type, v1, v2, v3
  [, scope])"，其中：

  * *addr_type* 取 "TIPC_ADDR_NAMESEQ"、"TIPC_ADDR_NAME" 或
    "TIPC_ADDR_ID" 中的一个。

  * *scope* 取 "TIPC_ZONE_SCOPE"、"TIPC_CLUSTER_SCOPE" 和
    "TIPC_NODE_SCOPE" 中的一个。

  * 如果 *addr_type* 为 "TIPC_ADDR_NAME"，那么 *v1* 是服务器类型，*v2*
    是端口标识符，*v3* 应为 0。

    如果 *addr_type* 为 "TIPC_ADDR_NAMESEQ"，那么 *v1* 是服务器类型，
    *v2* 是端口号下限，而 *v3* 是端口号上限。

    如果 *addr_type* 为 "TIPC_ADDR_ID"，那么 *v1* 是节点 (node)，*v2*
    是 ref，*v3* 应为 0。

* "AF_CAN" 地址族使用元组 "(interface, )"，其中 *interface* 是表示网络
  接口名称的字符串，如 "'can0'"。网络接口名 "''" 可以用于接收本族所有
  网络接口的数据包。

  * "CAN_ISOTP" 协议接受一个元组 "(interface, rx_addr, tx_addr)"，其中
    两个额外参数都是无符号长整数，都表示 CAN 标识符（标准或扩展标识符
    ）。

* "PF_SYSTEM" 协议簇的 "SYSPROTO_CONTROL" 协议接受一个字符串或元组
  "(id, unit)"。其中字符串是内核控件的名称，该控件使用动态分配的 ID。
  而如果 ID 和内核控件的单元 (unit) 编号都已知，或使用了已注册的 ID，
  可以采用元组。

  3.3 新版功能.

* "AF_BLUETOOTH" 支持以下协议和地址格式：

  * "BTPROTO_L2CAP" 接受 "(bdaddr, psm)"，其中 "bdaddr" 为字符串格式的
    蓝牙地址，"psm" 是一个整数。

  * "BTPROTO_RFCOMM" 接受 "(bdaddr, channel)"，其中 "bdaddr" 为字符串
    格式的蓝牙地址，"channel" 是一个整数。

  * "BTPROTO_HCI" 接受 "(device_id,)"，其中 "device_id" 为整数或字符串
    ，它表示接口对应的蓝牙地址（具体取决于你的系统，NetBSD 和
    DragonFlyBSD 需要蓝牙地址字符串，其他系统需要整数）。

    在 3.2 版更改: 添加了对 NetBSD 和 DragonFlyBSD 的支持。

  * "BTPROTO_SCO" 接受 "bdaddr"，其中 "bdaddr" 是 "bytes" 对象，其中含
    有字符串格式的蓝牙地址（如 "b'12:23:34:45:56:67'" ），FreeBSD 不支
    持此协议。

* "AF_ALG" 是一个仅 Linux 可用的、基于套接字的接口，用于连接内核加密算
  法。算法套接字可用包括 2 至 4 个元素的元组来配置 "(type, name [,
  feat [, mask]])"，其中：

  * *type* 是表示算法类型的字符串，如 "aead"、"hash"、"skcipher" 或
    "rng"。

  * *name* 是表示算法类型和操作模式的字符串，如 "sha256"、
    "hmac(sha256)"、"cbc(aes)" 或 "drbg_nopr_ctr_aes256"。

  * *feat* 和 *mask* 是无符号 32 位整数。

  Availability: Linux 2.6.38, some algorithm types require more recent
  Kernels.

  3.6 新版功能.

* "AF_VSOCK" 用于支持虚拟机与宿主机之间的通讯。该套接字用 "(CID,
  port)" 元组表示，其中 Context ID (CID) 和 port 都是整数。

  Availability: Linux >= 4.8 QEMU >= 2.8 ESX >= 4.0 ESX Workstation >=
  6.5.

  3.7 新版功能.

* "AF_PACKET" 是一个底层接口，直接连接至网卡。数据包使用元组 "(ifname,
  proto[, pkttype[, hatype[, addr]]])" 表示，其中：

  * *ifname* - 指定设备名称的字符串。

  * *proto* - 一个用网络字节序表示的整数，指定以太网协议版本号。

  * *pkttype* - 指定数据包类型的整数（可选）：

    * "PACKET_HOST" （默认） - 寻址到本地主机的数据包。

    * "PACKET_BROADCAST" - 物理层广播的数据包。

    * "PACKET_MULTIHOST" - 发送到物理层多播地址的数据包。

    * "PACKET_OTHERHOST" - 被（处于混杂模式的）网卡驱动捕获的、发送到
      其他主机的数据包。

    * "PACKET_OUTGOING" - 来自本地主机的、回环到一个套接字的数据包。

  * *hatype* - 可选整数，指定 ARP 硬件地址类型。

  * *addr* - 可选的类字节串对象，用于指定硬件物理地址，其解释取决于各
    设备。

* "AF_QIPCRTR" 是一个仅 Linux 可用的、基于套接字的接口，用于与高通平台
  中协处理器上运行的服务进行通信。该地址簇用一个 "(node, port)" 元组表
  示，其中 *node* 和 *port* 为非负整数。

  3.8 新版功能.

如果你在 IPv4/v6 套接字地址的 *host* 部分中使用了一个主机名，此程序可
能会表现不确定行为，因为 Python 使用 DNS 解析返回的第一个地址。套接字
地址在实际的 IPv4/v6 中以不同方式解析，根据 DNS 解析和/或 host 配置。
为了确定行为，在 *host* 部分中使用数字的地址。

所有的错误都抛出异常。对于无效的参数类型和内存溢出异常情况可能抛出普通
异常；从 Python 3.3 开始，与套接字或地址语义有关的错误抛出 "OSError"
或它的子类之一（常用 "socket.error"）。

可以用 "setblocking()" 设置非阻塞模式。一个基于超时的 generalization
通过 "settimeout()" 支持。


模块内容
========

"socket" 模块导出以下元素。


异常
----

exception socket.error

   一个被弃用的 "OSError" 的别名。

   在 3.3 版更改: 根据 **PEP 3151**，这个类是 "OSError" 的别名。

exception socket.herror

   "OSError" 的子类，本异常通常表示与地址相关的错误，比如那些在 POSIX
   C API 中使用了 *h_errno* 的函数，包括 "gethostbyname_ex()" 和
   "gethostbyaddr()"。附带的值是一对 "(h_errno, string)"，代表库调用返
   回的错误。*h_errno* 是一个数字，而 *string* 表示 *h_errno* 的描述，
   它们由 C 函数 "hstrerror()" 返回。

   在 3.3 版更改: 此类是 "OSError" 的子类。

exception socket.gaierror

   "OSError" 的子类，本异常来自 "getaddrinfo()" 和 "getnameinfo()"，表
   示与地址相关的错误。附带的值是一对 "(error, string)"，代表库调用返
   回的错误。*string* 表示 *error* 的描述，它由 C 函数
   "gai_strerror()" 返回。数字值 *error* 与本模块中定义的 "EAI_*" 常量
   之一匹配。

   在 3.3 版更改: 此类是 "OSError" 的子类。

exception socket.timeout

   "OSError" 的子类，当套接字发生超时，且事先已调用过 "settimeout()"
   （或隐式地通过 "setdefaulttimeout()" ）启用了超时，则会抛出此异常。
   附带的值是一个字符串，其值总是 "timed out"。

   在 3.3 版更改: 此类是 "OSError" 的子类。


常数
----

   AF_* 和 SOCK_* 常量现在都在 "AddressFamily" 和 "SocketKind" 这两个
   "IntEnum" 集合内。

   3.4 新版功能.

socket.AF_UNIX
socket.AF_INET
socket.AF_INET6

   这些常量表示地址（和协议）簇，用于 "socket()" 的第一个参数。如果
   "AF_UNIX" 常量未定义，即表示不支持该协议。不同系统可能会有更多其他
   常量可用。

socket.SOCK_STREAM
socket.SOCK_DGRAM
socket.SOCK_RAW
socket.SOCK_RDM
socket.SOCK_SEQPACKET

   这些常量表示套接字类型，用于 "socket()" 的第二个参数。不同系统可能
   会有更多其他常量可用。（一般只有 "SOCK_STREAM" 和 "SOCK_DGRAM" 可用
   ）

socket.SOCK_CLOEXEC
socket.SOCK_NONBLOCK

   这两个常量（如果已定义）可以与上述套接字类型结合使用，允许你设置这
   些原子性相关的 flags （从而避免可能的竞争条件和单独调用的需要）。

   参见: Secure File Descriptor Handling （安全地处理文件描述符） 提供了更
       详尽的解释。

   可用性： Linux >= 2.6.27。

   3.2 新版功能.

SO_*
socket.SOMAXCONN
MSG_*
SOL_*
SCM_*
IPPROTO_*
IPPORT_*
INADDR_*
IP_*
IPV6_*
EAI_*
AI_*
NI_*
TCP_*

   此列表内的许多常量，记载在 Unix 文档中的套接字和/或 IP 协议部分，同
   时也定义在本 socket 模块中。它们通常用于套接字对象的 "setsockopt()"
   和 "getsockopt()" 方法的参数中。在大多数情况下，仅那些在 Unix 头文
   件中有定义的符号会在本模块中定义，部分符号提供了默认值。

   在 3.6 版更改: 添加了 "SO_DOMAIN"、"SO_PROTOCOL"、"SO_PEERSEC"、
   "SO_PASSSEC"、"TCP_USER_TIMEOUT"、"TCP_CONGESTION"。

   在 3.6.5 版更改: 在 Windows 上，如果 Windows 运行时支持，则
   "TCP_FASTOPEN"、"TCP_KEEPCNT" 可用。

   在 3.7 版更改: 添加了 "TCP_NOTSENT_LOWAT"。在 Windows 上，如果
   Windows 运行时支持，则 "TCP_KEEPIDLE"、"TCP_KEEPINTVL" 可用。

socket.AF_CAN
socket.PF_CAN
SOL_CAN_*
CAN_*

   此列表内的许多常量，记载在 Linux 文档中，同时也定义在本 socket 模块
   中。

   可用性： Linux >= 2.6.25。

   3.3 新版功能.

socket.CAN_BCM
CAN_BCM_*

   CAN 协议簇内的 CAN_BCM 是广播管理器（Bbroadcast Manager -- BCM）协
   议，广播管理器常量在 Linux 文档中有所记载，在本 socket 模块中也有定
   义。

   可用性： Linux >= 2.6.25。

   注解:

     "CAN_BCM_CAN_FD_FRAME" 旗标仅在 Linux >= 4.8 时可用。

   3.4 新版功能.

socket.CAN_RAW_FD_FRAMES

   在 CAN_RAW 套接字中启用 CAN FD 支持，默认是禁用的。它使应用程序可以
   发送 CAN 和 CAN FD 帧。但是，从套接字读取时，也必须同时接受 CAN 和
   CAN FD 帧。

   此常量在 Linux 文档中有所记载。

   可用性： Linux >= 3.6。

   3.5 新版功能.

socket.CAN_ISOTP

   CAN 协议簇中的 CAN_ISOTP 就是 ISO-TP (ISO 15765-2) 协议。ISO-TP 常
   量在 Linux 文档中有所记载。

   可用性： Linux >= 2.6.25。

   3.7 新版功能.

socket.AF_PACKET
socket.PF_PACKET
PACKET_*

   此列表内的许多常量，记载在 Linux 文档中，同时也定义在本 socket 模块
   中。

   可用性： Linux >= 2.2。

socket.AF_RDS
socket.PF_RDS
socket.SOL_RDS
RDS_*

   此列表内的许多常量，记载在 Linux 文档中，同时也定义在本 socket 模块
   中。

   可用性： Linux >= 2.6.30。

   3.3 新版功能.

socket.SIO_RCVALL
socket.SIO_KEEPALIVE_VALS
socket.SIO_LOOPBACK_FAST_PATH
RCVALL_*

   Windows 的 WSAIoctl() 的常量。这些常量用于套接字对象的 "ioctl()" 方
   法的参数。

   在 3.6 版更改: 添加了 "SIO_LOOPBACK_FAST_PATH"。

TIPC_*

   TIPC 相关常量，与 C socket API 导出的常量一致。更多信息请参阅 TIPC
   文档。

socket.AF_ALG
socket.SOL_ALG
ALG_*

   用于 Linux 内核加密算法的常量。

   可用性： Linux >= 2.6.38。

   3.6 新版功能.

socket.AF_VSOCK
socket.IOCTL_VM_SOCKETS_GET_LOCAL_CID
VMADDR*
SO_VM*

   用于 Linux 宿主机/虚拟机通讯的常量。

   可用性： Linux >= 4.8。

   3.7 新版功能.

socket.AF_LINK

   可用性： BSD、OSX。

   3.4 新版功能.

socket.has_ipv6

   本常量为一个布尔值，该值指示当前平台是否支持 IPv6。

socket.BDADDR_ANY
socket.BDADDR_LOCAL

   这些是字符串常量，包含蓝牙地址，这些地址具有特殊含义。例如，当用
   "BTPROTO_RFCOMM" 指定绑定套接字时， "BDADDR_ANY" 表示“任何地址”。

socket.HCI_FILTER
socket.HCI_TIME_STAMP
socket.HCI_DATA_DIR

   与 "BTPROTO_HCI" 一起使用。 "HCI_FILTER" 在 NetBSD 或 DragonFlyBSD
   上不可用。 "HCI_TIME_STAMP" 和 "HCI_DATA_DIR" 在 FreeBSD、NetBSD 或
   DragonFlyBSD 上不可用。

socket.AF_QIPCRTR

   高通 IPC 路由协议的常数，用于与提供远程处理器的服务进行通信。

   可用性： Linux >= 4.7。


函数
----


创建套接字
~~~~~~~~~~

下列函数都能创建 套接字对象.

socket.socket(family=AF_INET, type=SOCK_STREAM, proto=0, fileno=None)

   使用给定的地址簇、套接字类型和协议号创建一个新的套接字。地址簇应为
   "AF_INET" （默认）、"AF_INET6"、"AF_UNIX"、"AF_CAN"、"AF_PACKET" 或
   "AF_RDS" 其中之一。套接字类型应为 "SOCK_STREAM" （默认）、
   "SOCK_DGRAM"、"SOCK_RAW" 或其他 "SOCK_" 常量之一。协议号通常为零，
   可以省略，或者在地址簇为 "AF_CAN" 的情况下，协议号应为 "CAN_RAW"、
   "CAN_BCM" 或 "CAN_ISOTP" 之一。

   如果指定了 *fileno*，那么将从这一指定的文件描述符中自动检测
   *family*、*type* 和 *proto* 的值。如果调用本函数时显式指定了
   *family*、*type* 或 *proto* 参数，可以覆盖自动检测的值。这只会影响
   Python 表示诸如 "socket.getpeername()" 一类函数的返回值的方式，而不
   影响实际的操作系统资源。与 "socket.fromfd()" 不同，*fileno* 将返回
   原先的套接字，而不是复制出新的套接字。这有助于在分离的套接字上调用
   "socket.close()" 来关闭它。

   新创建的套接字是 不可继承的。

   引发一个 审计事件 "socket.__new__" 附带参数 "self"、"family"、
   "type"、"protocol"。

   在 3.3 版更改: 添加了 AF_CAN 簇。添加了 AF_RDS 簇。

   在 3.4 版更改: 添加了 CAN_BCM 协议。

   在 3.4 版更改: 返回的套接字现在是不可继承的。

   在 3.7 版更改: 添加了 CAN_ISOTP 协议。

   在 3.7 版更改: 当将 "SOCK_NONBLOCK" 或 "SOCK_CLOEXEC" 标志位应用于
   *type* 上时，它们会被清除，且 "socket.type" 反映不出它们。但它们仍
   将传递给底层系统的 *socket()* 调用。因此，

      sock = socket.socket(
          socket.AF_INET,
          socket.SOCK_STREAM | socket.SOCK_NONBLOCK)

   仍将在支持 "SOCK_NONBLOCK" 的系统上创建一个非阻塞的套接字，但是
   "sock.type" 会被置为 "socket.SOCK_STREAM"。

socket.socketpair([family[, type[, proto]]])

   构建一对已连接的套接字对象，使用给定的地址簇、套接字类型和协议号。
   地址簇、套接字类型和协议号与上述 "socket()" 函数相同。默认地址簇为
   "AF_UNIX" （需要当前平台支持，不支持则默认为 "AF_INET" ）。

   新创建的套接字都是 不可继承的。

   在 3.2 版更改: 现在，返回的套接字对象支持全部套接字 API，而不是全部
   API 的一个子集。

   在 3.4 版更改: 返回的套接字现在都是不可继承的。

   在 3.5 版更改: 添加了 Windows 支持。

socket.create_connection(address[, timeout[, source_address]])

   连接到一个 TCP 服务，该服务正在侦听 Internet *address* （用二元组
   "(host, port)" 表示）。连接后返回套接字对象。这是比
   "socket.connect()" 更高级的函数：如果 *host* 是非数字主机名，它将尝
   试从 "AF_INET" 和 "AF_INET6" 解析它，然后依次尝试连接到所有可能的地
   址，直到连接成功。这使得编写兼容 IPv4 和 IPv6 的客户端变得容易。

   传入可选参数 *timeout* 可以在套接字实例上设置超时（在尝试连接前）。
   如果未提供 *timeout*，则使用由 "getdefaulttimeout()" 返回的全局默认
   超时设置。

   如果提供了 *source_address*，它必须为二元组 "(host, port)"，以便套
   接字在连接之前绑定为其源地址。如果 host 或 port 分别为 '' 或 0，则
   使用操作系统默认行为。

   在 3.2 版更改: 添加了 *source_address*。

socket.create_server(address, *, family=AF_INET, backlog=None, reuse_port=False, dualstack_ipv6=False)

   便捷函数，创建绑定到 *address* （二元组 "(host, port)" ）的 TCP 套
   接字，返回套接字对象。

   *family* 应设置为 "AF_INET" 或 "AF_INET6"。*backlog* 是传递给
   "socket.listen()" 的队列大小，当它为 "0" 则表示默认的合理值。
   *reuse_port* 表示是否设置 "SO_REUSEPORT" 套接字选项。

   如果 *dualstack_ipv6* 为 true 且平台支持，则套接字能接受 IPv4 和
   IPv6 连接，否则将抛出 "ValueError" 异常。大多数 POSIX 平台和
   Windows 应该支持此功能。启用此功能后，"socket.getpeername()" 在进行
   IPv4 连接时返回的地址将是一个（映射到 IPv4 的）IPv6 地址。在默认启
   用该功能的平台上（如 Linux），如果 *dualstack_ipv6* 为 false，即显
   式禁用此功能。该参数可以与 "has_dualstack_ipv6()" 结合使用：

      import socket

      addr = ("", 8080)  # all interfaces, port 8080
      if socket.has_dualstack_ipv6():
          s = socket.create_server(addr, family=socket.AF_INET6, dualstack_ipv6=True)
      else:
          s = socket.create_server(addr)

   注解:

     在 POSIX 平台上，设置 "SO_REUSEADDR" 套接字选项是为了立即重用以前
     绑定在同一 *address* 上并保持 TIME_WAIT 状态的套接字。

   3.8 新版功能.

socket.has_dualstack_ipv6()

   如果平台支持创建 IPv4 和 IPv6 连接都可以处理的 TCP 套接字，则返回
   "True"。

   3.8 新版功能.

socket.fromfd(fd, family, type, proto=0)

   复制文件描述符 *fd* （一个由文件对象的 "fileno()" 方法返回的整数）
   ，然后从结果中构建一个套接字对象。地址簇、套接字类型和协议号与上述
   "socket()" 函数相同。文件描述符应指向一个套接字，但不会专门去检查——
   如果文件描述符是无效的，则对该对象的后续操作可能会失败。本函数很少
   用到，但是在将套接字作为标准输入或输出传递给程序（如 Unix inet 守护
   程序启动的服务器）时，可以使用本函数获取或设置套接字选项。套接字将
   处于阻塞模式。

   新创建的套接字是 不可继承的。

   在 3.4 版更改: 返回的套接字现在是不可继承的。

socket.fromshare(data)

   根据 "socket.share()" 方法获得的数据实例化套接字。套接字将处于阻塞
   模式。

   可用性: Windows。

   3.3 新版功能.

socket.SocketType

   这是一个 Python 类型对象，表示套接字对象的类型。它等同于
   "type(socket(...))"。


其他功能
~~~~~~~~

"socket" 模块还提供多种网络相关服务：

socket.close(fd)

   关闭一个套接字文件描述符。它类似于 "os.close()"，但专用于套接字。在
   某些平台上（特别是在 Windows 上），"os.close()" 对套接字文件描述符
   无效。

   3.7 新版功能.

socket.getaddrinfo(host, port, family=0, type=0, proto=0, flags=0)

   将 *host*/*port* 参数转换为 5 元组的序列，其中包含创建（连接到某服
   务的）套接字所需的所有参数。*host* 是域名，是字符串格式的 IPv4/v6
   地址或 "None"。*port* 是字符串格式的服务名称，如 "'http'" 、端口号
   （数字）或 "None"。传入 "None" 作为 *host* 和 *port* 的值，相当于将
   "NULL" 传递给底层 C API。

   可以指定 *family*、*type* 和 *proto* 参数，以缩小返回的地址列表。向
   这些参数分别传入 0 表示保留全部结果范围。*flags* 参数可以是 "AI_*"
   常量中的一个或多个，它会影响结果的计算和返回。例如，
   "AI_NUMERICHOST" 会禁用域名解析，此时如果 *host* 是域名，则会抛出错
   误。

   本函数返回一个列表，其中的 5 元组具有以下结构：

   "(family, type, proto, canonname, sockaddr)"

   在这些元组中，*family*、*type*、*proto* 都是整数，可以用于传递给
   "socket()" 函数。如果 *flags* 参数有一部分是 "AI_CANONNAME"，那么
   *canonname* 将是表示 *host* 的规范名称的字符串。否则 *canonname* 将
   为空。*sockaddr* 是一个表示套接字地址的元组，具体格式取决于返回的
   *family* （对于 "AF_INET"，是一个 "(address, port)" 二元组，对于
   "AF_INET6"，是一个 "(address, port, flow info, scope id)" 四元组）
   ，可以用于传递给 "socket.connect()" 方法。

   引发一个 审计事件 "socket.getaddrinfo" 附带参数 "host"、"port"、
   "family"、"type"、"protocol"。

   下面的示例获取了 TCP 连接地址信息，假设该连接通过 80 端口连接至
   "example.org" （如果系统未启用 IPv6，则结果可能会不同）:

      >>> socket.getaddrinfo("example.org", 80, proto=socket.IPPROTO_TCP)
      [(<AddressFamily.AF_INET6: 10>, <SocketType.SOCK_STREAM: 1>,
       6, '', ('2606:2800:220:1:248:1893:25c8:1946', 80, 0, 0)),
       (<AddressFamily.AF_INET: 2>, <SocketType.SOCK_STREAM: 1>,
       6, '', ('93.184.216.34', 80))]

   在 3.2 版更改: 现在可以使用关键字参数的形式来传递参数。

   在 3.7 版更改: 对于 IPv6 多播地址，表示地址的字符串将不包含
   "%scope" 部分。

socket.getfqdn([name])

   返回 *name* 的全限定域名 (Fully Qualified Domain Name -- FQDN)。如
   果 *name* 省略或为空，则将其解释为本地主机。为了查找全限定名称，首
   先将检查由 "gethostbyaddr()" 返回的主机名，然后是主机的别名（如果存
   在）。选中第一个包含句点的名字。如果无法获取全限定域名，则返回由
   "gethostname()" 返回的主机名。

socket.gethostbyname(hostname)

   将主机名转换为 IPv4 地址格式。IPv4 地址以字符串格式返回，如
   "'100.50.200.5'"。如果主机名本身是 IPv4 地址，则原样返回。更完整的
   接口请参考 "gethostbyname_ex()"。 "gethostbyname()" 不支持 IPv6 名
   称解析，应使用 "getaddrinfo()" 来支持 IPv4/v6 双协议栈。

   引发一个 审计事件 "socket.gethostbyname"，附带参数 "hostname"。

socket.gethostbyname_ex(hostname)

   将主机名转换为 IPv4 地址格式的扩展接口。返回三元组 "(hostname,
   aliaslist, ipaddrlist)"，其中 *hostname* 是响应给定 *ip_address* 的
   主要主机名，*aliaslist* 是相同地址的其他可用主机名的列表（可能为空
   ），而 *ipaddrlist* 是 IPv4 地址列表，包含相同主机名、相同接口的不
   同地址（通常是一个地址，但不总是如此）。"gethostbyname_ex()" 不支持
   IPv6 名称解析，应使用 "getaddrinfo()" 来支持 IPv4/v6 双协议栈。

   引发一个 审计事件 "socket.gethostbyname"，附带参数 "hostname"。

socket.gethostname()

   返回一个字符串，包含当前正在运行 Python 解释器的机器的主机名。

   引发一个 审计事件 "socket.gethostname"，没有附带参数。

   注意： "gethostname()" 并不总是返回全限定域名，必要的话请使用
   "getfqdn()"。

socket.gethostbyaddr(ip_address)

   返回三元组 "(hostname, aliaslist, ipaddrlist)"，其中 *hostname* 是
   响应给定 *ip_address* 的主要主机名，*aliaslist* 是相同地址的其他可
   用主机名的列表（可能为空），而 *ipaddrlist* 是 IPv4/v6 地址列表，包
   含相同主机名、相同接口的不同地址（很可能仅包含一个地址）。要查询全
   限定域名，请使用函数 "getfqdn()"。"gethostbyaddr()" 支持 IPv4 和
   IPv6。

   引发一个 审计事件 "socket.gethostbyaddr"，附带参数 "ip_address"。

socket.getnameinfo(sockaddr, flags)

   将套接字地址 *sockaddr* 转换为二元组 "(host, port)"。*host* 的形式
   可能是全限定域名，或是由数字表示的地址，具体取决于 *flags* 的设置。
   同样，*port* 可以包含字符串格式的端口名称或数字格式的端口号。

   对于 IPv6 地址，如果 *sockaddr* 包含有意义的 *scopeid*，则将
   "%scope" 附加到 host 部分。这种情况通常发生在多播地址上。

   关于 *flags* 的更多信息可参阅 *getnameinfo(3)*。

   引发一个 审计事件 "socket.getnameinfo"，附带参数 "sockaddr"。

socket.getprotobyname(protocolname)

   将 Internet 协议名称（如 "'icmp'" ）转换为常量，该常量适用于
   "socket()" 函数的第三个（可选）参数。通常只有在 "raw" 模式
   ("SOCK_RAW") 中打开的套接字才需要使用该常量。在正常的套接字模式下，
   协议省略或为零时，会自动选择正确的协议。

socket.getservbyname(servicename[, protocolname])

   将 Internet 服务名称和协议名称转换为该服务的端口号。协议名称是可选
   的，如果提供的话应为 "'tcp'" 或 "'udp'"，否则将匹配出所有协议。

   引发一个 审计事件 "socket.getservbyname"，附带参数 "servicename"、
   "protocolname"。

socket.getservbyport(port[, protocolname])

   将 Internet 端口号和协议名称转换为该服务的服务名称。协议名称是可选
   的，如果提供的话应为 "'tcp'" 或 "'udp'"，否则将匹配出所有协议。

   引发一个 审计事件 "socket.getservbyport"，附带参数 "port"、
   "protocolname"。

socket.ntohl(x)

   将 32 位正整数从网络字节序转换为主机字节序。在主机字节序与网络字节
   序相同的计算机上，这是一个空操作。字节序不同将执行 4 字节交换操作。

socket.ntohs(x)

   将 16 位正整数从网络字节序转换为主机字节序。在主机字节序与网络字节
   序相同的计算机上，这是一个空操作。字节序不同将执行 2 字节交换操作。

   3.7 版后已移除: 如果 *x* 不符合 16 位无符号整数，但符合 C 语言的正
   整数，则它会被静默截断为 16 位无符号整数。此静默截断功能已弃用，未
   来版本的 Python 将对此抛出异常。

socket.htonl(x)

   将 32 位正整数从主机字节序转换为网络字节序。在主机字节序与网络字节
   序相同的计算机上，这是一个空操作。字节序不同将执行 4 字节交换操作。

socket.htons(x)

   将 16 位正整数从主机字节序转换为网络字节序。在主机字节序与网络字节
   序相同的计算机上，这是一个空操作。字节序不同将执行 2 字节交换操作。

   3.7 版后已移除: 如果 *x* 不符合 16 位无符号整数，但符合 C 语言的正
   整数，则它会被静默截断为 16 位无符号整数。此静默截断功能已弃用，未
   来版本的 Python 将对此抛出异常。

socket.inet_aton(ip_string)

   将 IPv4 地址从点分十进制字符串格式（如 '123.45.67.89' ）转换为 32
   位压缩二进制格式，转换后为字节对象，长度为四个字符。与那些使用标准
   C 库，且需要 "struct in_addr" 类型的对象的程序交换信息时，本函数很
   有用。 该类型即本函数返回的 32 位压缩二进制的 C 类型。

   "inet_aton()" 也接受句点数少于三的字符串，详情请参阅 Unix 手册
   *inet(3)*。

   如果传入本函数的 IPv4 地址字符串无效，则抛出 "OSError"。注意，具体
   什么样的地址有效取决于 "inet_aton()" 的底层 C 实现。

   "inet_aton()" 不支持 IPv6，在 IPv4/v6 双协议栈下应使用
   "inet_pton()" 来代替。

socket.inet_ntoa(packed_ip)

   将 32 位压缩 IPv4 地址（一个 *类字节对象*，长 4 个字节）转换为标准
   的点分十进制字符串形式（如 '123.45.67.89' ）。与那些使用标准 C 库，
   且需要 "struct in_addr" 类型的对象的程序交换信息时，本函数很有用。
   该类型即本函数参数中的 32 位压缩二进制数据的 C 类型。

   如果传入本函数的字节序列长度不是 4 个字节，则抛出 "OSError"。
   "inet_ntoa()" 不支持 IPv6，在 IPv4/v6 双协议栈下应使用
   "inet_ntop()" 来代替。

   在 3.5 版更改: 现在支持可写的 *字节类对象*。

socket.inet_pton(address_family, ip_string)

   将特定地址簇的 IP 地址（字符串）转换为压缩二进制格式。当库或网络协
   议需要接受 "struct in_addr" 类型的对象（类似 "inet_aton()" ）或
   "struct in6_addr" 类型的对象时，"inet_pton()" 很有用。

   目前 *address_family* 支持 "AF_INET" 和 "AF_INET6"。如果 IP 地址字
   符串 *ip_string* 无效，则抛出 "OSError"。注意，具体什么地址有效取决
   于 *address_family* 的值和 "inet_pton()" 的底层实现。

   可用性： Unix（可能非所有平台都可用）、Windows。

   在 3.4 版更改: 添加了 Windows 支持

socket.inet_ntop(address_family, packed_ip)

   将压缩 IP 地址（一个 *类字节对象*，数个字节长）转换为标准的、特定地
   址簇的字符串形式（如 "'7.10.0.5'" 或 "'5aef:2b::8'" ）。当库或网络
   协议返回 "struct in_addr" 类型的对象（类似 "inet_ntoa()" ）或
   "struct in6_addr" 类型的对象时，"inet_ntop()" 很有用。

   目前 *address_family* 支持 "AF_INET" 和 "AF_INET6"。如果字节对象
   *packed_ip* 与指定的地址簇长度不符，则抛出 "ValueError"。针对
   "inet_ntop()" 调用的错误则抛出 "OSError"。

   可用性： Unix（可能非所有平台都可用）、Windows。

   在 3.4 版更改: 添加了 Windows 支持

   在 3.5 版更改: 现在支持可写的 *字节类对象*。

socket.CMSG_LEN(length)

   返回给定 *length* 所关联数据的辅助数据项的总长度（不带尾部填充）。
   此值通常用作 "recvmsg()" 接收一个辅助数据项的缓冲区大小，但是 **RFC
   3542** 要求可移植应用程序使用 "CMSG_SPACE()"，以此将尾部填充的空间
   计入，即使该项在缓冲区的最后。如果 *length* 超出允许范围，则抛出
   "OverflowError"。

   可用性： 大多数 Unix 平台，其他平台也可能可用。

   3.3 新版功能.

socket.CMSG_SPACE(length)

   返回 "recvmsg()" 所需的缓冲区大小，以接收给定 *length* 所关联数据的
   辅助数据项，带有尾部填充。接收多个项目所需的缓冲区空间是关联数据长
   度的 "CMSG_SPACE()" 值的总和。如果 *length* 超出允许范围，则抛出
   "OverflowError"。

   请注意，某些系统可能支持辅助数据，但不提供本函数。还需注意，如果使
   用本函数的结果来设置缓冲区大小，可能无法精确限制可接收的辅助数据量
   ，因为可能会有其他数据写入尾部填充区域。

   可用性： 大多数 Unix 平台，其他平台也可能可用。

   3.3 新版功能.

socket.getdefaulttimeout()

   返回用于新套接字对象的默认超时（以秒为单位的浮点数）。值 "None" 表
   示新套接字对象没有超时。首次导入 socket 模块时，默认值为 "None"。

socket.setdefaulttimeout(timeout)

   设置用于新套接字对象的默认超时（以秒为单位的浮点数）。首次导入
   socket 模块时，默认值为 "None"。可能的取值及其各自的含义请参阅
   "settimeout()"。

socket.sethostname(name)

   将计算机的主机名设置为 *name*。如果权限不足将抛出 "OSError"。

   引发一个 审计事件 "socket.sethostname"，附带参数 "name"。

   Availability: Unix.

   3.3 新版功能.

socket.if_nameindex()

   返回一个列表，包含网络接口（网卡）信息二元组（整数索引，名称字符串
   ）。系统调用失败则抛出 "OSError"。

   可用性: Unix, Windows。

   3.3 新版功能.

   在 3.8 版更改: 添加了 Windows 支持。

socket.if_nametoindex(if_name)

   返回网络接口名称相对应的索引号。如果没有所给名称的接口，则抛出
   "OSError"。

   可用性: Unix, Windows。

   3.3 新版功能.

   在 3.8 版更改: 添加了 Windows 支持。

socket.if_indextoname(if_index)

   返回网络接口索引号相对应的接口名称。如果没有所给索引号的接口，则抛
   出 "OSError"。

   可用性: Unix, Windows。

   3.3 新版功能.

   在 3.8 版更改: 添加了 Windows 支持。


套接字对象
==========

套接字对象具有以下方法。除了 "makefile()"，其他都与套接字专用的 Unix
系统调用相对应。

在 3.2 版更改: 添加了对 *上下文管理器* 协议的支持。退出上下文管理器与
调用 "close()" 等效。

socket.accept()

   接受一个连接。此 scoket 必须绑定到一个地址上并且监听连接。返回值是
   一个 "(conn, address)" 对，其中 *conn* 是一个 *新* 的套接字对象，用
   于在此连接上收发数据，*address* 是连接另一端的套接字所绑定的地址。

   新创建的套接字是 不可继承的。

   在 3.4 版更改: 该套接字现在是不可继承的。

   在 3.5 版更改: 如果系统调用被中断，但信号处理程序没有触发异常，此方
   法现在会重试系统调用，而不是触发 "InterruptedError" 异常 (原因详见
   **PEP 475**)。

socket.bind(address)

   将套接字绑定到 *address*。套接字必须尚未绑定。（ *address* 的格式取
   决于地址簇 —— 参见上文）

   引发一个 审计事件 "socket.bind"，附带参数 "self"、"address"。

socket.close()

   Mark the socket closed.  The underlying system resource (e.g. a
   file descriptor) is also closed when all file objects from
   "makefile()" are closed.  Once that happens, all future operations
   on the socket object will fail. The remote end will receive no more
   data (after queued data is flushed).

   Sockets are automatically closed when they are garbage-collected,
   but it is recommended to "close()" them explicitly, or to use a
   "with" statement around them.

   在 3.6 版更改: "OSError" is now raised if an error occurs when the
   underlying "close()" call is made.

   注解:

     "close()" releases the resource associated with a connection but
     does not necessarily close the connection immediately.  If you
     want to close the connection in a timely fashion, call
     "shutdown()" before "close()".

socket.connect(address)

   Connect to a remote socket at *address*. (The format of *address*
   depends on the address family --- see above.)

   If the connection is interrupted by a signal, the method waits
   until the connection completes, or raise a "socket.timeout" on
   timeout, if the signal handler doesn't raise an exception and the
   socket is blocking or has a timeout. For non-blocking sockets, the
   method raises an "InterruptedError" exception if the connection is
   interrupted by a signal (or the exception raised by the signal
   handler).

   引发一个 审计事件 "socket.connect"，附带参数 "self"、"address"。

   在 3.5 版更改: The method now waits until the connection completes
   instead of raising an "InterruptedError" exception if the
   connection is interrupted by a signal, the signal handler doesn't
   raise an exception and the socket is blocking or has a timeout (see
   the **PEP 475** for the rationale).

socket.connect_ex(address)

   Like "connect(address)", but return an error indicator instead of
   raising an exception for errors returned by the C-level "connect()"
   call (other problems, such as "host not found," can still raise
   exceptions).  The error indicator is "0" if the operation
   succeeded, otherwise the value of the "errno" variable.  This is
   useful to support, for example, asynchronous connects.

   引发一个 审计事件 "socket.connect"，附带参数 "self"、"address"。

socket.detach()

   Put the socket object into closed state without actually closing
   the underlying file descriptor.  The file descriptor is returned,
   and can be reused for other purposes.

   3.2 新版功能.

socket.dup()

   Duplicate the socket.

   新创建的套接字是 不可继承的。

   在 3.4 版更改: 该套接字现在是不可继承的。

socket.fileno()

   Return the socket's file descriptor (a small integer), or -1 on
   failure. This is useful with "select.select()".

   Under Windows the small integer returned by this method cannot be
   used where a file descriptor can be used (such as "os.fdopen()").
   Unix does not have this limitation.

socket.get_inheritable()

   Get the inheritable flag of the socket's file descriptor or
   socket's handle: "True" if the socket can be inherited in child
   processes, "False" if it cannot.

   3.4 新版功能.

socket.getpeername()

   Return the remote address to which the socket is connected.  This
   is useful to find out the port number of a remote IPv4/v6 socket,
   for instance. (The format of the address returned depends on the
   address family --- see above.)  On some systems this function is
   not supported.

socket.getsockname()

   Return the socket's own address.  This is useful to find out the
   port number of an IPv4/v6 socket, for instance. (The format of the
   address returned depends on the address family --- see above.)

socket.getsockopt(level, optname[, buflen])

   Return the value of the given socket option (see the Unix man page
   *getsockopt(2)*).  The needed symbolic constants ("SO_*" etc.) are
   defined in this module.  If *buflen* is absent, an integer option
   is assumed and its integer value is returned by the function.  If
   *buflen* is present, it specifies the maximum length of the buffer
   used to receive the option in, and this buffer is returned as a
   bytes object.  It is up to the caller to decode the contents of the
   buffer (see the optional built-in module "struct" for a way to
   decode C structures encoded as byte strings).

socket.getblocking()

   Return "True" if socket is in blocking mode, "False" if in non-
   blocking.

   This is equivalent to checking "socket.gettimeout() == 0".

   3.7 新版功能.

socket.gettimeout()

   Return the timeout in seconds (float) associated with socket
   operations, or "None" if no timeout is set.  This reflects the last
   call to "setblocking()" or "settimeout()".

socket.ioctl(control, option)

   Platform:
      Windows

   The "ioctl()" method is a limited interface to the WSAIoctl system
   interface.  Please refer to the Win32 documentation for more
   information.

   On other platforms, the generic "fcntl.fcntl()" and "fcntl.ioctl()"
   functions may be used; they accept a socket object as their first
   argument.

   Currently only the following control codes are supported:
   "SIO_RCVALL", "SIO_KEEPALIVE_VALS", and "SIO_LOOPBACK_FAST_PATH".

   在 3.6 版更改: 添加了 "SIO_LOOPBACK_FAST_PATH"。

socket.listen([backlog])

   Enable a server to accept connections.  If *backlog* is specified,
   it must be at least 0 (if it is lower, it is set to 0); it
   specifies the number of unaccepted connections that the system will
   allow before refusing new connections. If not specified, a default
   reasonable value is chosen.

   在 3.5 版更改: The *backlog* parameter is now optional.

socket.makefile(mode='r', buffering=None, *, encoding=None, errors=None, newline=None)

   Return a *file object* associated with the socket.  The exact
   returned type depends on the arguments given to "makefile()".
   These arguments are interpreted the same way as by the built-in
   "open()" function, except the only supported *mode* values are
   "'r'" (default), "'w'" and "'b'".

   The socket must be in blocking mode; it can have a timeout, but the
   file object's internal buffer may end up in an inconsistent state
   if a timeout occurs.

   Closing the file object returned by "makefile()" won't close the
   original socket unless all other file objects have been closed and
   "socket.close()" has been called on the socket object.

   注解:

     On Windows, the file-like object created by "makefile()" cannot
     be used where a file object with a file descriptor is expected,
     such as the stream arguments of "subprocess.Popen()".

socket.recv(bufsize[, flags])

   Receive data from the socket.  The return value is a bytes object
   representing the data received.  The maximum amount of data to be
   received at once is specified by *bufsize*.  See the Unix manual
   page *recv(2)* for the meaning of the optional argument *flags*; it
   defaults to zero.

   注解:

     For best match with hardware and network realities, the value of
     *bufsize* should be a relatively small power of 2, for example,
     4096.

   在 3.5 版更改: 如果系统调用被中断，但信号处理程序没有触发异常，此方
   法现在会重试系统调用，而不是触发 "InterruptedError" 异常 (原因详见
   **PEP 475**)。

socket.recvfrom(bufsize[, flags])

   Receive data from the socket.  The return value is a pair "(bytes,
   address)" where *bytes* is a bytes object representing the data
   received and *address* is the address of the socket sending the
   data.  See the Unix manual page *recv(2)* for the meaning of the
   optional argument *flags*; it defaults to zero. (The format of
   *address* depends on the address family --- see above.)

   在 3.5 版更改: 如果系统调用被中断，但信号处理程序没有触发异常，此方
   法现在会重试系统调用，而不是触发 "InterruptedError" 异常 (原因详见
   **PEP 475**)。

   在 3.7 版更改: For multicast IPv6 address, first item of *address*
   does not contain "%scope" part anymore. In order to get full IPv6
   address use "getnameinfo()".

socket.recvmsg(bufsize[, ancbufsize[, flags]])

   Receive normal data (up to *bufsize* bytes) and ancillary data from
   the socket.  The *ancbufsize* argument sets the size in bytes of
   the internal buffer used to receive the ancillary data; it defaults
   to 0, meaning that no ancillary data will be received.  Appropriate
   buffer sizes for ancillary data can be calculated using
   "CMSG_SPACE()" or "CMSG_LEN()", and items which do not fit into the
   buffer might be truncated or discarded.  The *flags* argument
   defaults to 0 and has the same meaning as for "recv()".

   The return value is a 4-tuple: "(data, ancdata, msg_flags,
   address)".  The *data* item is a "bytes" object holding the non-
   ancillary data received.  The *ancdata* item is a list of zero or
   more tuples "(cmsg_level, cmsg_type, cmsg_data)" representing the
   ancillary data (control messages) received: *cmsg_level* and
   *cmsg_type* are integers specifying the protocol level and
   protocol-specific type respectively, and *cmsg_data* is a "bytes"
   object holding the associated data.  The *msg_flags* item is the
   bitwise OR of various flags indicating conditions on the received
   message; see your system documentation for details. If the
   receiving socket is unconnected, *address* is the address of the
   sending socket, if available; otherwise, its value is unspecified.

   On some systems, "sendmsg()" and "recvmsg()" can be used to pass
   file descriptors between processes over an "AF_UNIX" socket.  When
   this facility is used (it is often restricted to "SOCK_STREAM"
   sockets), "recvmsg()" will return, in its ancillary data, items of
   the form "(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)", where *fds*
   is a "bytes" object representing the new file descriptors as a
   binary array of the native C "int" type.  If "recvmsg()" raises an
   exception after the system call returns, it will first attempt to
   close any file descriptors received via this mechanism.

   Some systems do not indicate the truncated length of ancillary data
   items which have been only partially received.  If an item appears
   to extend beyond the end of the buffer, "recvmsg()" will issue a
   "RuntimeWarning", and will return the part of it which is inside
   the buffer provided it has not been truncated before the start of
   its associated data.

   On systems which support the "SCM_RIGHTS" mechanism, the following
   function will receive up to *maxfds* file descriptors, returning
   the message data and a list containing the descriptors (while
   ignoring unexpected conditions such as unrelated control messages
   being received).  See also "sendmsg()".

      import socket, array

      def recv_fds(sock, msglen, maxfds):
          fds = array.array("i")   # Array of ints
          msg, ancdata, flags, addr = sock.recvmsg(msglen, socket.CMSG_LEN(maxfds * fds.itemsize))
          for cmsg_level, cmsg_type, cmsg_data in ancdata:
              if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                  # Append data, ignoring any truncated integers at the end.
                  fds.frombytes(cmsg_data[:len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
          return msg, list(fds)

   可用性： 大多数 Unix 平台，其他平台也可能可用。

   3.3 新版功能.

   在 3.5 版更改: 如果系统调用被中断，但信号处理程序没有触发异常，此方
   法现在会重试系统调用，而不是触发 "InterruptedError" 异常 (原因详见
   **PEP 475**)。

socket.recvmsg_into(buffers[, ancbufsize[, flags]])

   Receive normal data and ancillary data from the socket, behaving as
   "recvmsg()" would, but scatter the non-ancillary data into a series
   of buffers instead of returning a new bytes object.  The *buffers*
   argument must be an iterable of objects that export writable
   buffers (e.g. "bytearray" objects); these will be filled with
   successive chunks of the non-ancillary data until it has all been
   written or there are no more buffers.  The operating system may set
   a limit ("sysconf()" value "SC_IOV_MAX") on the number of buffers
   that can be used.  The *ancbufsize* and *flags* arguments have the
   same meaning as for "recvmsg()".

   The return value is a 4-tuple: "(nbytes, ancdata, msg_flags,
   address)", where *nbytes* is the total number of bytes of non-
   ancillary data written into the buffers, and *ancdata*, *msg_flags*
   and *address* are the same as for "recvmsg()".

   示例:

      >>> import socket
      >>> s1, s2 = socket.socketpair()
      >>> b1 = bytearray(b'----')
      >>> b2 = bytearray(b'0123456789')
      >>> b3 = bytearray(b'--------------')
      >>> s1.send(b'Mary had a little lamb')
      22
      >>> s2.recvmsg_into([b1, memoryview(b2)[2:9], b3])
      (22, [], 0, None)
      >>> [b1, b2, b3]
      [bytearray(b'Mary'), bytearray(b'01 had a 9'), bytearray(b'little lamb---')]

   可用性： 大多数 Unix 平台，其他平台也可能可用。

   3.3 新版功能.

socket.recvfrom_into(buffer[, nbytes[, flags]])

   Receive data from the socket, writing it into *buffer* instead of
   creating a new bytestring.  The return value is a pair "(nbytes,
   address)" where *nbytes* is the number of bytes received and
   *address* is the address of the socket sending the data.  See the
   Unix manual page *recv(2)* for the meaning of the optional argument
   *flags*; it defaults to zero.  (The format of *address* depends on
   the address family --- see above.)

socket.recv_into(buffer[, nbytes[, flags]])

   Receive up to *nbytes* bytes from the socket, storing the data into
   a buffer rather than creating a new bytestring.  If *nbytes* is not
   specified (or 0), receive up to the size available in the given
   buffer.  Returns the number of bytes received.  See the Unix manual
   page *recv(2)* for the meaning of the optional argument *flags*; it
   defaults to zero.

socket.send(bytes[, flags])

   Send data to the socket.  The socket must be connected to a remote
   socket.  The optional *flags* argument has the same meaning as for
   "recv()" above. Returns the number of bytes sent. Applications are
   responsible for checking that all data has been sent; if only some
   of the data was transmitted, the application needs to attempt
   delivery of the remaining data. For further information on this
   topic, consult the 套接字编程指南.

   在 3.5 版更改: 如果系统调用被中断，但信号处理程序没有触发异常，此方
   法现在会重试系统调用，而不是触发 "InterruptedError" 异常 (原因详见
   **PEP 475**)。

socket.sendall(bytes[, flags])

   Send data to the socket.  The socket must be connected to a remote
   socket.  The optional *flags* argument has the same meaning as for
   "recv()" above. Unlike "send()", this method continues to send data
   from *bytes* until either all data has been sent or an error
   occurs.  "None" is returned on success.  On error, an exception is
   raised, and there is no way to determine how much data, if any, was
   successfully sent.

   在 3.5 版更改: The socket timeout is no more reset each time data
   is sent successfully. The socket timeout is now the maximum total
   duration to send all data.

   在 3.5 版更改: 如果系统调用被中断，但信号处理程序没有触发异常，此方
   法现在会重试系统调用，而不是触发 "InterruptedError" 异常 (原因详见
   **PEP 475**)。

socket.sendto(bytes, address)
socket.sendto(bytes, flags, address)

   Send data to the socket.  The socket should not be connected to a
   remote socket, since the destination socket is specified by
   *address*.  The optional *flags* argument has the same meaning as
   for "recv()" above.  Return the number of bytes sent. (The format
   of *address* depends on the address family --- see above.)

   引发一个 审计事件 "socket.sendto"，附带参数 "self"、"address"。

   在 3.5 版更改: 如果系统调用被中断，但信号处理程序没有触发异常，此方
   法现在会重试系统调用，而不是触发 "InterruptedError" 异常 (原因详见
   **PEP 475**)。

socket.sendmsg(buffers[, ancdata[, flags[, address]]])

   Send normal and ancillary data to the socket, gathering the non-
   ancillary data from a series of buffers and concatenating it into a
   single message.  The *buffers* argument specifies the non-ancillary
   data as an iterable of *bytes-like objects* (e.g. "bytes" objects);
   the operating system may set a limit ("sysconf()" value
   "SC_IOV_MAX") on the number of buffers that can be used.  The
   *ancdata* argument specifies the ancillary data (control messages)
   as an iterable of zero or more tuples "(cmsg_level, cmsg_type,
   cmsg_data)", where *cmsg_level* and *cmsg_type* are integers
   specifying the protocol level and protocol-specific type
   respectively, and *cmsg_data* is a bytes-like object holding the
   associated data.  Note that some systems (in particular, systems
   without "CMSG_SPACE()") might support sending only one control
   message per call.  The *flags* argument defaults to 0 and has the
   same meaning as for "send()".  If *address* is supplied and not
   "None", it sets a destination address for the message.  The return
   value is the number of bytes of non-ancillary data sent.

   The following function sends the list of file descriptors *fds*
   over an "AF_UNIX" socket, on systems which support the "SCM_RIGHTS"
   mechanism.  See also "recvmsg()".

      import socket, array

      def send_fds(sock, msg, fds):
          return sock.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))])

   可用性： 大多数 Unix 平台，其他平台也可能可用。

   引发一个 审计事件 "socket.sendmsg"，附带参数 "self"、"address"。

   3.3 新版功能.

   在 3.5 版更改: 如果系统调用被中断，但信号处理程序没有触发异常，此方
   法现在会重试系统调用，而不是触发 "InterruptedError" 异常 (原因详见
   **PEP 475**)。

socket.sendmsg_afalg([msg], *, op[, iv[, assoclen[, flags]]])

   Specialized version of "sendmsg()" for "AF_ALG" socket. Set mode,
   IV, AEAD associated data length and flags for "AF_ALG" socket.

   可用性： Linux >= 2.6.38。

   3.6 新版功能.

socket.sendfile(file, offset=0, count=None)

   Send a file until EOF is reached by using high-performance
   "os.sendfile" and return the total number of bytes which were sent.
   *file* must be a regular file object opened in binary mode. If
   "os.sendfile" is not available (e.g. Windows) or *file* is not a
   regular file "send()" will be used instead. *offset* tells from
   where to start reading the file. If specified, *count* is the total
   number of bytes to transmit as opposed to sending the file until
   EOF is reached. File position is updated on return or also in case
   of error in which case "file.tell()" can be used to figure out the
   number of bytes which were sent. The socket must be of
   "SOCK_STREAM" type. Non-blocking sockets are not supported.

   3.5 新版功能.

socket.set_inheritable(inheritable)

   Set the inheritable flag of the socket's file descriptor or
   socket's handle.

   3.4 新版功能.

socket.setblocking(flag)

   Set blocking or non-blocking mode of the socket: if *flag* is
   false, the socket is set to non-blocking, else to blocking mode.

   This method is a shorthand for certain "settimeout()" calls:

   * "sock.setblocking(True)" is equivalent to "sock.settimeout(None)"

   * "sock.setblocking(False)" is equivalent to "sock.settimeout(0.0)"

   在 3.7 版更改: The method no longer applies "SOCK_NONBLOCK" flag on
   "socket.type".

socket.settimeout(value)

   Set a timeout on blocking socket operations.  The *value* argument
   can be a nonnegative floating point number expressing seconds, or
   "None". If a non-zero value is given, subsequent socket operations
   will raise a "timeout" exception if the timeout period *value* has
   elapsed before the operation has completed.  If zero is given, the
   socket is put in non-blocking mode. If "None" is given, the socket
   is put in blocking mode.

   For further information, please consult the notes on socket
   timeouts.

   在 3.7 版更改: The method no longer toggles "SOCK_NONBLOCK" flag on
   "socket.type".

socket.setsockopt(level, optname, value: int)

socket.setsockopt(level, optname, value: buffer)

socket.setsockopt(level, optname, None, optlen: int)

   Set the value of the given socket option (see the Unix manual page
   *setsockopt(2)*).  The needed symbolic constants are defined in the
   "socket" module ("SO_*" etc.).  The value can be an integer, "None"
   or a *bytes-like object* representing a buffer. In the later case
   it is up to the caller to ensure that the bytestring contains the
   proper bits (see the optional built-in module "struct" for a way to
   encode C structures as bytestrings). When *value* is set to "None",
   *optlen* argument is required. It's equivalent to call
   "setsockopt()" C function with "optval=NULL" and "optlen=optlen".

   在 3.5 版更改: 现在支持可写的 *字节类对象*。

   在 3.6 版更改: setsockopt(level, optname, None, optlen: int) form
   added.

socket.shutdown(how)

   Shut down one or both halves of the connection.  If *how* is
   "SHUT_RD", further receives are disallowed.  If *how* is "SHUT_WR",
   further sends are disallowed.  If *how* is "SHUT_RDWR", further
   sends and receives are disallowed.

socket.share(process_id)

   Duplicate a socket and prepare it for sharing with a target
   process.  The target process must be provided with *process_id*.
   The resulting bytes object can then be passed to the target process
   using some form of interprocess communication and the socket can be
   recreated there using "fromshare()". Once this method has been
   called, it is safe to close the socket since the operating system
   has already duplicated it for the target process.

   可用性: Windows。

   3.3 新版功能.

Note that there are no methods "read()" or "write()"; use "recv()" and
"send()" without *flags* argument instead.

Socket objects also have these (read-only) attributes that correspond
to the values given to the "socket" constructor.

socket.family

   The socket family.

socket.type

   The socket type.

socket.proto

   The socket protocol.


Notes on socket timeouts
========================

A socket object can be in one of three modes: blocking, non-blocking,
or timeout.  Sockets are by default always created in blocking mode,
but this can be changed by calling "setdefaulttimeout()".

* In *blocking mode*, operations block until complete or the system
  returns an error (such as connection timed out).

* In *non-blocking mode*, operations fail (with an error that is
  unfortunately system-dependent) if they cannot be completed
  immediately: functions from the "select" can be used to know when
  and whether a socket is available for reading or writing.

* In *timeout mode*, operations fail if they cannot be completed
  within the timeout specified for the socket (they raise a "timeout"
  exception) or if the system returns an error.

注解:

  At the operating system level, sockets in *timeout mode* are
  internally set in non-blocking mode.  Also, the blocking and timeout
  modes are shared between file descriptors and socket objects that
  refer to the same network endpoint. This implementation detail can
  have visible consequences if e.g. you decide to use the "fileno()"
  of a socket.


Timeouts and the "connect" method
---------------------------------

The "connect()" operation is also subject to the timeout setting, and
in general it is recommended to call "settimeout()" before calling
"connect()" or pass a timeout parameter to "create_connection()".
However, the system network stack may also return a connection timeout
error of its own regardless of any Python socket timeout setting.


Timeouts and the "accept" method
--------------------------------

If "getdefaulttimeout()" is not "None", sockets returned by the
"accept()" method inherit that timeout.  Otherwise, the behaviour
depends on settings of the listening socket:

* if the listening socket is in *blocking mode* or in *timeout mode*,
  the socket returned by "accept()" is in *blocking mode*;

* if the listening socket is in *non-blocking mode*, whether the
  socket returned by "accept()" is in blocking or non-blocking mode is
  operating system-dependent.  If you want to ensure cross-platform
  behaviour, it is recommended you manually override this setting.


示例
====

Here are four minimal example programs using the TCP/IP protocol: a
server that echoes all data that it receives back (servicing only one
client), and a client using it.  Note that a server must perform the
sequence "socket()", "bind()", "listen()", "accept()" (possibly
repeating the "accept()" to service more than one client), while a
client only needs the sequence "socket()", "connect()".  Also note
that the server does not "sendall()"/"recv()" on the socket it is
listening on but on the new socket returned by "accept()".

The first two examples support IPv4 only.

   # Echo server program
   import socket

   HOST = ''                 # Symbolic name meaning all available interfaces
   PORT = 50007              # Arbitrary non-privileged port
   with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
       s.bind((HOST, PORT))
       s.listen(1)
       conn, addr = s.accept()
       with conn:
           print('Connected by', addr)
           while True:
               data = conn.recv(1024)
               if not data: break
               conn.sendall(data)

   # Echo client program
   import socket

   HOST = 'daring.cwi.nl'    # The remote host
   PORT = 50007              # The same port as used by the server
   with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
       s.connect((HOST, PORT))
       s.sendall(b'Hello, world')
       data = s.recv(1024)
   print('Received', repr(data))

The next two examples are identical to the above two, but support both
IPv4 and IPv6. The server side will listen to the first address family
available (it should listen to both instead). On most of IPv6-ready
systems, IPv6 will take precedence and the server may not accept IPv4
traffic. The client side will try to connect to the all addresses
returned as a result of the name resolution, and sends traffic to the
first one connected successfully.

   # Echo server program
   import socket
   import sys

   HOST = None               # Symbolic name meaning all available interfaces
   PORT = 50007              # Arbitrary non-privileged port
   s = None
   for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC,
                                 socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
       af, socktype, proto, canonname, sa = res
       try:
           s = socket.socket(af, socktype, proto)
       except OSError as msg:
           s = None
           continue
       try:
           s.bind(sa)
           s.listen(1)
       except OSError as msg:
           s.close()
           s = None
           continue
       break
   if s is None:
       print('could not open socket')
       sys.exit(1)
   conn, addr = s.accept()
   with conn:
       print('Connected by', addr)
       while True:
           data = conn.recv(1024)
           if not data: break
           conn.send(data)

   # Echo client program
   import socket
   import sys

   HOST = 'daring.cwi.nl'    # The remote host
   PORT = 50007              # The same port as used by the server
   s = None
   for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC, socket.SOCK_STREAM):
       af, socktype, proto, canonname, sa = res
       try:
           s = socket.socket(af, socktype, proto)
       except OSError as msg:
           s = None
           continue
       try:
           s.connect(sa)
       except OSError as msg:
           s.close()
           s = None
           continue
       break
   if s is None:
       print('could not open socket')
       sys.exit(1)
   with s:
       s.sendall(b'Hello, world')
       data = s.recv(1024)
   print('Received', repr(data))

The next example shows how to write a very simple network sniffer with
raw sockets on Windows. The example requires administrator privileges
to modify the interface:

   import socket

   # the public network interface
   HOST = socket.gethostbyname(socket.gethostname())

   # create a raw socket and bind it to the public interface
   s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
   s.bind((HOST, 0))

   # Include IP headers
   s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)

   # receive all packages
   s.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)

   # receive a package
   print(s.recvfrom(65565))

   # disabled promiscuous mode
   s.ioctl(socket.SIO_RCVALL, socket.RCVALL_OFF)

The next example shows how to use the socket interface to communicate
to a CAN network using the raw socket protocol. To use CAN with the
broadcast manager protocol instead, open a socket with:

   socket.socket(socket.AF_CAN, socket.SOCK_DGRAM, socket.CAN_BCM)

After binding ("CAN_RAW") or connecting ("CAN_BCM") the socket, you
can use the "socket.send()", and the "socket.recv()" operations (and
their counterparts) on the socket object as usual.

This last example might require special privileges:

   import socket
   import struct


   # CAN frame packing/unpacking (see 'struct can_frame' in <linux/can.h>)

   can_frame_fmt = "=IB3x8s"
   can_frame_size = struct.calcsize(can_frame_fmt)

   def build_can_frame(can_id, data):
       can_dlc = len(data)
       data = data.ljust(8, b'\x00')
       return struct.pack(can_frame_fmt, can_id, can_dlc, data)

   def dissect_can_frame(frame):
       can_id, can_dlc, data = struct.unpack(can_frame_fmt, frame)
       return (can_id, can_dlc, data[:can_dlc])


   # create a raw socket and bind it to the 'vcan0' interface
   s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
   s.bind(('vcan0',))

   while True:
       cf, addr = s.recvfrom(can_frame_size)

       print('Received: can_id=%x, can_dlc=%x, data=%s' % dissect_can_frame(cf))

       try:
           s.send(cf)
       except OSError:
           print('Error sending CAN frame')

       try:
           s.send(build_can_frame(0x01, b'\x01\x02\x03'))
       except OSError:
           print('Error sending CAN frame')

Running an example several times with too small delay between
executions, could lead to this error:

   OSError: [Errno 98] Address already in use

This is because the previous execution has left the socket in a
"TIME_WAIT" state, and can't be immediately reused.

There is a "socket" flag to set, in order to prevent this,
"socket.SO_REUSEADDR":

   s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
   s.bind((HOST, PORT))

the "SO_REUSEADDR" flag tells the kernel to reuse a local socket in
"TIME_WAIT" state, without waiting for its natural timeout to expire.

参见:

  For an introduction to socket programming (in C), see the following
  papers:

  * *An Introductory 4.3BSD Interprocess Communication Tutorial*, by
    Stuart Sechrest

  * *An Advanced 4.3BSD Interprocess Communication Tutorial*, by
    Samuel J.  Leffler et al,

  both in the UNIX Programmer's Manual, Supplementary Documents 1
  (sections PS1:7 and PS1:8).  The platform-specific reference
  material for the various socket-related system calls are also a
  valuable source of information on the details of socket semantics.
  For Unix, refer to the manual pages; for Windows, see the WinSock
  (or Winsock 2) specification.  For IPv6-ready APIs, readers may want
  to refer to **RFC 3493** titled Basic Socket Interface Extensions
  for IPv6.
