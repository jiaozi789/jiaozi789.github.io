---
title: "计算机网络基础02-linux虚拟网络隔离（网桥bridge，路由，虚拟网卡veth）"
date: 2025-09-18T16:55:17+08:00
weight: 1
# bookComments: false
# bookSearchExclude: false
---

# 简单概述
通过linux的虚拟网络隔离，来模拟熟悉交换机，路由器，网卡,网桥等物理设备间的关系和区别。
以下几个概念：
- network namespace：实现网络虚拟化的重要功能，它能创建多个隔离的网络空间，它们有独自的网络栈信息。不管是虚拟机还是容器，运行的时候仿佛自己就在独立的网络中。
- veth：VETH(Virtual Ethernet )是Linux提供的另外一种特殊的网络设备，中文称为虚拟网卡接口。它总是成对出现，要创建就创建一个pair。一个Pair中的veth就像一个网络线缆的两个端点，数据从一个端点进入，必然从另外一个端点流出。每个veth都可以被赋予IP地址，并参与三层网络路由过程，可以实现不同netns之间网络通信。
- 物理网卡是服务器上实际的网络接口设备：物理网卡指的是服务器上实际的网络接口设备，通过ip link可查看设备，其中eth0一般是第一个网卡，ethn表示第n个,通过ip address查看ip地址。
```
liaomin@DESKTOP-FSEDE3P:~$ ip link
16: eth0: <BROADCAST,MULTICAST,UP> mtu 1500 group default qlen 1
    link/ether 78:24:af:33:df:11
20: eth1: <BROADCAST,MULTICAST,UP> mtu 1500 group default qlen 1
    link/ether 7a:15:df:c5:e6:d5
1: lo: <LOOPBACK,UP> mtu 1500 group default qlen 1
    link/loopback 00:00:00:00:00:00
```
- 网桥：是一种虚拟设备，可以将 Linux 内部多个网络接口连接起来，一个网络接口接收到网络数据包后，会复制到其他网络接口中，Bridge 是二层设备，仅用来处理二层的通讯。Bridge 使用 MAC 地址表来决定怎么转发帧（Frame）。Bridge 会从 host 之间的通讯数据包中学习 MAC 地址。
![在这里插入图片描述](/docs/images/content/devops/networking/basic/net_basic_02.md.images/b2d31af95dacbcfa5bd171f3d7223640.png)
如上图所示，当网络接口A接收到数据包后，网桥 会将数据包复制并且发送给连接到 网桥 的其他网络接口（如上图中的网卡B和网卡C）。

- 子网卡：子网卡在这里并不是实际上的网络接口设备，但是可以作为网络接口在系统中出现，如eth0:1、eth1:2这种网络接口。它们必须要依赖于物理网卡，虽然可以与物理网卡的网络接口同时在系统中存在并使用不同的IP地址，而且也拥有它们自己的网络接口配置文件。但是当所依赖的物理网卡不启用时（Down状态）这些子网卡也将一同不能工作，可通过ipvlan或者ifconfig来创建。
- 路由（Routing）：路由是指从一个设备（一般指路由器）的接口上接收到数据包，依据设备所既定的某些规则，将数据包转发到其它接口的 “过程”。路由工作在 OSI 参考模型第三层——网络层的数据包转发设备。路由器通过转发数据包来实现网络互连，正常情况下，路由器不会修改数据包的源地址和目标地址，只是路由器启用NAT功能后，会将IP地址和TCP端口绑定重新定义一个出口IP地址和新的端口号（端口号可能不变，也可能变，家用路由器上网源地址的端口号是要变的）。
- 网关：一般是路由器的ip地址，比如需要访问另外一个网络的ip时，可以先将数据包丢给路由器，路由器，可以将包转换到另外的网卡或者通过nat转换后在发送到另外网卡，linux本身有个ip_forward功能可以在网卡间转发数据包（路由器功能），只是没有nat功能而已，可以通过iptables来实现nat功能。

# 环境模拟
![在这里插入图片描述](/docs/images/content/devops/networking/basic/net_basic_02.md.images/e3e303ebff6e31243cf41be784a4cf64.png)

研究这个可以直接使用自己的window，安装个虚拟机安装个ubuntu然后在ubuntu系统中使用ip命令隔离网络来研究网络概念。
1. window主机有本地连接（ip地址：192.168.20.48）网关是192.168.20.1。
2. window主机安装了hyer-v虚拟机，带有一个defaultswitch网关，我这里ip地址是172.168.111.1/24
3. 在hyver-v虚拟机上有一台ubuntu虚拟机，网卡为:172.168.111.2
# network ns 
## 网络虚拟化
network namespace 是实现网络虚拟化的重要功能，它能创建多个隔离的网络空间，它们有独自的网络栈信息。不管是虚拟机还是容器，运行的时候仿佛自己就在独立的网络中。这篇文章介绍 network namespace 的基本概念和用法，network namespace 是 linux 内核提供的功能，这篇文章借助 ip 命令来完成各种操作。ip 命令来自于 iproute2 安装包，一般系统会默认安装。

ip 命令管理的功能很多， 和 network namespace 有关的操作都是在子命令 ip netns 下进行的，可以通过 ip netns help` 查看所有操作的帮助信息。

默认情况下，使用 ip netns 是没有网络 namespace 的，所以 ip netns ls（简写ip net） 命令看不到任何输出。
```
[root@localhost ~]# ip netns help
Usage: ip netns list
       ip netns add NAME
       ip netns delete NAME
       ip netns identify PID
       ip netns pids NAME
       ip netns exec NAME cmd ...
       ip netns monitor
[root@localhost ~]# ip netns ls
```
创建 network namespace 也非常简单，直接使用 ip netns add 后面跟着要创建的 namespace 名称。如果相同名字的 namespace 已经存在，命令会报 Cannot create namespace 的错误。

```
[root@localhost ~]# ip netns add net1
[root@localhost ~]# ip netns ls
net1
```
ip netns 命令创建的 network namespace 会出现在 /var/run/netns/ 目录下，如果需要管理其他不是 ip netns 创建的 network namespace，只要在这个目录下创建一个指向对应 network namespace 文件的链接就行
有了自己创建的 network namespace，我们还需要看看它里面有哪些东西。对于每个 network namespace 来说，它会有自己独立的网卡、路由表、ARP 表、iptables 等和网络相关的资源。ip 命令提供了 ip netns exec 子命令可以在对应的 network namespace 中执行命令，比如我们要看一下这个 network namespace 中有哪些网卡。更棒的是，要执行的可以是任何命令，不只是和网络相关的（当然，和网络无关命令执行的结果和在外部执行没有区别）。比如下面例子中，执行 bash 命令了之后，后面所有的命令都是在这个 network namespace 中执行的，好处是不用每次执行命令都要把 ip netns exec NAME 补全，缺点是你无法清楚知道自己当前所在的 shell，容易混淆。

```
[root@localhost ~]# ip netns exec net1 ip addr
1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
[root@localhost ~]# ip netns exec net1 bash
[root@localhost ~]# ip addr
1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
[root@localhost ~]# exit
exit
```
每个 namespace 在创建的时候会自动创建一个 lo 的 interface，它的作用和 linux 系统中默认看到的 lo 一样，都是为了实现 loopback 通信。如果希望 lo 能工作，不要忘记启用它：
```
[root@localhost ~]# ip netns exec net1 ip link set lo up
```
><b><font color=red>默认情况下，network namespace 是不能和主机网络，或者其他 network namespace 通信的。</font></b>

### 网络间通讯
#### 虚拟网卡
##### veth pair
###### 虚拟网络互通
有了不同 network namespace 之后，也就有了网络的隔离，但是如果它们之间没有办法通信，也没有实际用处。要把两个网络连接起来，linux 提供了 veth pair 。可以把 veth pair 当做是双向的 pipe（管道），从一个方向发送的网络数据，可以直接被另外一端接收到；或者也可以想象成两个 namespace 直接通过一个特殊的虚拟网卡连接起来，可以直接通信。
使用上面提到的方法，我们再创建另外一个 network namespace，这里我们使用 net1 和 net2 两个名字。

```
ip net add net1 && ip net add net2
```
![在这里插入图片描述](/docs/images/content/devops/networking/basic/net_basic_02.md.images/e66e4eb7109c21d63e344888baf47a41.png)


我们可以使用 ip link add type veth 来创建一对 veth pair 出来，需要记住的是 veth pair 无法单独存在，删除其中一个（ip link delete veth0），另一个也会自动消失。
通过命令创建的veth默认是挂载在宿主机上,创建两对veth，一个用于在net1和net2间连接，另外一个用于net1和宿主机进行连接

```
root@liaomin-Virtual-Machine:/home/liaomin# ip link add veth0 type veth peer name veth1 && ip link add veth2 type veth peer name veth3
root@liaomin-Virtual-Machine:/home/liaomin# ip link
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT group default qlen 1000
    link/ether 00:15:5d:04:11:01 brd ff:ff:ff:ff:ff:ff
7: veth1@veth0: <BROADCAST,MULTICAST,M-DOWN> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether 2a:1d:9f:9e:9f:23 brd ff:ff:ff:ff:ff:ff
8: veth0@veth1: <BROADCAST,MULTICAST,M-DOWN> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether 62:79:b5:a4:4a:ff brd ff:ff:ff:ff:ff:ff
9: veth3@veth2: <BROADCAST,MULTICAST,M-DOWN> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether 2a:92:b7:67:0f:13 brd ff:ff:ff:ff:ff:ff
10: veth2@veth3: <BROADCAST,MULTICAST,M-DOWN> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether 9a:9d:09:4a:72:c9 brd ff:ff:ff:ff:ff:ff
```
注意 默认veth网卡是down模式，需要启动
将网卡添加到对应网络中。

```
ip link set veth0 netns net1 && ip link set veth1 netns net2 && ip link set veth2 netns net1 
```
检验下 宿主机只有veth3
```
root@liaomin-Virtual-Machine:/home/liaomin# ip link
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT group default qlen 1000
    link/ether 00:15:5d:04:11:01 brd ff:ff:ff:ff:ff:ff
9: veth3@if10: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether 2a:92:b7:67:0f:13 brd ff:ff:ff:ff:ff:ff link-netns net1
```
net1下 有veth0和veth2
```
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net1 ip link
1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
8: veth0@if7: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether 62:79:b5:a4:4a:ff brd ff:ff:ff:ff:ff:ff link-netns net2
10: veth2@if9: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether 9a:9d:09:4a:72:c9 brd ff:ff:ff:ff:ff:ff link-netnsid 0
```
net2下有veth1

```
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net2 ip link
1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
7: veth1@if8: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN mode DEFAULT group default qlen 1000
    link/ether 2a:1d:9f:9e:9f:23 brd ff:ff:ff:ff:ff:ff link-netns net1
```
分别设置ip和启动网卡

```
ip net exec net1 ip addr add 192.167.2.1/24 dev veth0  && ip net exec net1 ip link set veth0 up &&
ip net exec net2 ip addr add 192.167.2.2/24 dev veth1  && ip net exec net2 ip link set veth1 up &&
ip net exec net1 ip addr add 192.167.3.1/24 dev veth2  && ip net exec net1 ip link set veth2 up &&
ip addr add 192.167.3.2/24 dev veth3 &&  ip link set veth3 up
```
检验ip地址

```
root@liaomin-Virtual-Machine:/home/liaomin# ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 00:15:5d:04:11:01 brd ff:ff:ff:ff:ff:ff
    inet 172.168.111.2/24 brd 172.168.111.255 scope global noprefixroute eth0
       valid_lft forever preferred_lft forever
    inet6 fe80::5efd:a114:2485:3e10/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
9: veth3@if10: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
    link/ether 2a:92:b7:67:0f:13 brd ff:ff:ff:ff:ff:ff link-netns net1
    inet 192.167.3.2/24 scope global veth3
       valid_lft forever preferred_lft forever
    inet6 fe80::2892:b7ff:fe67:f13/64 scope link 
       valid_lft forever preferred_lft forever
```
net1下
```
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net1 ip addr
1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
8: veth0@if7: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
    link/ether 62:79:b5:a4:4a:ff brd ff:ff:ff:ff:ff:ff link-netns net2
    inet 192.167.2.1/24 scope global veth0
       valid_lft forever preferred_lft forever
    inet6 fe80::6079:b5ff:fea4:4aff/64 scope link 
       valid_lft forever preferred_lft forever
10: veth2@if9: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
    link/ether 9a:9d:09:4a:72:c9 brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 192.167.3.1/24 scope global veth2
       valid_lft forever preferred_lft forever
    inet6 fe80::989d:9ff:fe4a:72c9/64 scope link 
       valid_lft forever preferred_lft forever
```
测试网络情况
1. 宿主机ping veth2（通）和veth1（不通）
```
root@liaomin-Virtual-Machine:/home/liaomin# ping 192.167.3.1
PING 192.167.3.1 (192.167.3.1) 56(84) bytes of data.
64 比特，来自 192.167.3.1: icmp_seq=1 ttl=64 时间=0.047 毫秒
64 比特，来自 192.167.3.1: icmp_seq=2 ttl=64 时间=0.032 毫秒
^C
--- 192.167.3.1 ping 统计 ---
已发送 2 个包， 已接收 2 个包, 0% 包丢失, 耗时 1018 毫秒
rtt min/avg/max/mdev = 0.032/0.039/0.047/0.007 ms
root@liaomin-Virtual-Machine:/home/liaomin# ping 192.167.2.1
PING 192.167.2.1 (192.167.2.1) 56(84) bytes of data.
^C
--- 192.167.2.1 ping 统计 ---
已发送 2 个包， 已接收 0 个包, 100% 包丢失, 耗时 1019 毫秒
```
2. 在net1中访问宿主机veth3网卡（通）和eth0网卡（不通），以及net2 veth1网卡（通）

```
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net1 ping 192.167.2.2
PING 192.167.2.2 (192.167.2.2) 56(84) bytes of data.
64 比特，来自 192.167.2.2: icmp_seq=1 ttl=64 时间=0.042 毫秒
64 比特，来自 192.167.2.2: icmp_seq=2 ttl=64 时间=0.038 毫秒
^C
--- 192.167.2.2 ping 统计 ---
已发送 2 个包， 已接收 2 个包, 0% 包丢失, 耗时 1008 毫秒
rtt min/avg/max/mdev = 0.038/0.040/0.042/0.002 ms
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net1 ping 192.167.3.2
PING 192.167.3.2 (192.167.3.2) 56(84) bytes of data.
64 比特，来自 192.167.3.2: icmp_seq=1 ttl=64 时间=0.020 毫秒
64 比特，来自 192.167.3.2: icmp_seq=2 ttl=64 时间=0.044 毫秒
^C
--- 192.167.3.2 ping 统计 ---
已发送 2 个包， 已接收 2 个包, 0% 包丢失, 耗时 1024 毫秒
rtt min/avg/max/mdev = 0.020/0.032/0.044/0.012 ms
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net1 ping 172.168.111.2
ping: connect: 网络不可达
```
结论：
![在这里插入图片描述](/docs/images/content/devops/networking/basic/net_basic_02.md.images/a77d9cd9bfea97bcc0d75a0abdce0338.png)
1. 只有veth对对应的ip才能互通，net1和net2有veth0和veth1能访问192.167.2.0/24互通，但是都不能访问宿主机的eth0网卡。
2. net1和宿主机间只能通过veth2和veth3访问192.167.3.0/24互通，其他网卡ip不能互通。

###### ip互通（路由）
实现不同网络间不同网段ip可以互通，可以路由转发的功能实现所有ip互通，这里可以学习下路由的相关知识。
其实linux系统本身就是一个路由器，同一个主机多个网卡间进行数据包的转发，可通过系统参数net.ipv4.ip_forward控制，默认为0表示关闭。
> net1网络veth0接受到一个目的ip是192.167.3.1的报文，因为是在内部网络，添加路由后可直接访问，如果访问的是3.2另一个主机的地址，需要通过veth2中转发出，ip_forward=0则丢弃不会转发，=0则转发到veth2网卡，丢到其他主机。
> 同时要主机跨主机间数据包中转，需要双向路由，不然数据过去了是回不来的。

<font color=red>**打开ip_foward**</font>
在运行时修改（重启后失效）
```
sysctl -w net.ipv4.ip_forward=1
```
永久生效
vi /etc/sysctl.conf
修改或新增sysctl -w net.ipv4.ip_forward=1
执行命令加载文件生效
```
sysctl -p
```
<font color=red>**宿主机访问net1所有ip**</font>
当前状况下veth2ip:192.167.3.1通，veth0:192.167.2.1不通
宿主机因为能访问192.167.3.1，添加路由，访问192.167.2.0/24通过网关192.167.3.1，经过网卡veth3
> via网关指定的ip 一定要是可以访问的才能指定。
```
ip route add 192.167.2.0/24 via 192.167.3.1 dev veth3
```
查看路由
```
root@liaomin-Virtual-Machine:/home/liaomin# ip route add 192.167.2.0/24 via 192.167.3.1 dev veth3
root@liaomin-Virtual-Machine:/home/liaomin# ip route
default via 172.168.111.1 dev eth0 proto static metric 100 
169.254.0.0/16 dev eth0 scope link metric 1000 
172.168.111.0/24 dev eth0 proto kernel scope link src 172.168.111.2 metric 100 
192.167.2.0/24 via 192.167.3.1 dev veth3 
192.167.3.0/24 dev veth3 proto kernel scope link src 192.167.3.2
```
加上路由后，宿主机两个ip都通

```
root@liaomin-Virtual-Machine:/home/liaomin# ping 192.167.2.1
PING 192.167.2.1 (192.167.2.1) 56(84) bytes of data.
64 比特，来自 192.167.2.1: icmp_seq=1 ttl=64 时间=0.028 毫秒
64 比特，来自 192.167.2.1: icmp_seq=2 ttl=64 时间=0.033 毫秒
^C
--- 192.167.2.1 ping 统计 ---
已发送 2 个包， 已接收 2 个包, 0% 包丢失, 耗时 1027 毫秒
rtt min/avg/max/mdev = 0.028/0.030/0.033/0.002 ms
root@liaomin-Virtual-Machine:/home/liaomin# ping 192.167.3.1
PING 192.167.3.1 (192.167.3.1) 56(84) bytes of data.
64 比特，来自 192.167.3.1: icmp_seq=1 ttl=64 时间=0.019 毫秒
64 比特，来自 192.167.3.1: icmp_seq=2 ttl=64 时间=0.029 毫秒
```
关闭ip_foward测试下，依然能通说明，说明主机内部网络中转是和ip_forward无关

<font color=red>**宿主机通过net1访问net2ip**</font>
还是前面的路由，先打开ip_forward。宿主机尝试ping 192.167.2.2，明显不通
但是在net1网络中是可以通192.167.2.2，因为有veth
尝试在net2中抓包,发现只有request没有reply
```
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net2 tcpdump -nn -i veth1 icmp
tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on veth1, link-type EN10MB (Ethernet), capture size 262144 bytes
^C10:59:38.879329 IP 192.167.3.2 > 192.167.2.2: ICMP echo request, id 25, seq 1, length 64
10:59:39.888796 IP 192.167.3.2 > 192.167.2.2: ICMP echo request, id 25, seq 2, length 64
10:59:40.912715 IP 192.167.3.2 > 192.167.2.2: ICMP echo request, id 25, seq 3, length 64
10:59:41.936728 IP 192.167.3.2 > 192.167.2.2: ICMP echo request, id 25, seq 4, length 64
10:59:42.960757 IP 192.167.3.2 > 192.167.2.2: ICMP echo request, id 25, seq 5, length 64
```
说明来自宿主机192.167.3.2的包已经到了，但是宿主机因为接受不到回应。
关闭ip_forward（注意是net1是中转的，要关他）,在尝试抓包

```
ip net exec net1 sysctl -w net.ipv4.ip_forward=0  
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net1 sysctl -a | grep ip_forward      
net.ipv4.ip_forward = 0
net.ipv4.ip_forward_update_priority = 1
net.ipv4.ip_forward_use_pmtu = 0
```
发现无法抓取到包，证明了ip_forward功能
```
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net2 tcpdump -nn -i veth1 icmp
tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on veth1, link-type EN10MB (Ethernet), capture size 262144 bytes
^C
0 packets captured
0 packets received by filter
0 packets dropped by kernel
```
添加net2回程路由

```
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net2 ip route add 192.167.3.0/24 via 192.167.2.1 dev veth1 
root@liaomin-Virtual-Machine:/home/liaomin# ping 192.167.2.2
PING 192.167.2.2 (192.167.2.2) 56(84) bytes of data.
64 比特，来自 192.167.2.2: icmp_seq=1 ttl=63 时间=0.062 毫秒
64 比特，来自 192.167.2.2: icmp_seq=2 ttl=63 时间=0.043 毫秒
```
net2抓包，可以看到replay

```
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net2 tcpdump -nn -i veth1 icmp
tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on veth1, link-type EN10MB (Ethernet), capture size 262144 bytes
^C11:12:40.738563 IP 192.167.3.2 > 192.167.2.2: ICMP echo request, id 38, seq 1, length 64
11:12:40.738603 IP 192.167.2.2 > 192.167.3.2: ICMP echo reply, id 38, seq 1, length 64
11:12:41.744679 IP 192.167.3.2 > 192.167.2.2: ICMP echo request, id 38, seq 2, length 64
11:12:41.744714 IP 192.167.2.2 > 192.167.3.2: ICMP echo reply, id 38, seq 2, length 64
11:12:42.768699 IP 192.167.3.2 > 192.167.2.2: ICMP echo request, id 38, seq 3, length 64
11:12:42.768732 IP 192.167.2.2 > 192.167.3.2: ICMP echo reply, id 38, seq 3, length 64
11:12:43.792688 IP 192.167.3.2 > 192.167.2.2: ICMP echo request, id 38, seq 4, length 64
11:12:43.792722 IP 192.167.2.2 > 192.167.3.2: ICMP echo reply, id 38, seq 4, length 64
```
<font color=red>**window主机访问net2所有ip**</font>
![在这里插入图片描述](/docs/images/content/devops/networking/basic/net_basic_02.md.images/2e375dfd9d20b1759acf7670873ef508.png)

window下添加路由访问192.167.2.0/24和192.167.3.0/24设置网关为eth0ip 172.168.111.2，注意linux宿主机打开ip_forward

```
管理员权限执行
route add 192.167.3.0 mask 255.255.255.0 172.168.111.2
route add 192.167.2.0 mask 255.255.255.0 172.168.111.2
```
此时ping 192.167.3.2能通，但是3.1不通，数据包在net1可以抓到，因为没有配置回程路由说以无法访问，数据包原地址为172.168.111.1，

```
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net1 tcpdump -nn -i veth2  icmp
tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on veth2, link-type EN10MB (Ethernet), capture size 262144 bytes
^C15:28:10.774530 IP 172.168.111.1 > 192.167.3.1: ICMP echo request, id 1, seq 141, length 40
```
在net1上添加回程路由

```
ip net exec net1 ip route add 172.168.111.0/24 via 192.167.3.2 dev veth2
```
window上在ping，正常通了
```
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net1 tcpdump -nn -i veth2  icmp                                            
tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on veth2, link-type EN10MB (Ethernet), capture size 262144 bytes
^C15:32:26.718218 IP 172.168.111.1 > 192.167.3.1: ICMP echo request, id 1, seq 142, length 40
15:32:26.718261 IP 192.167.3.1 > 172.168.111.1: ICMP echo reply, id 1, seq 142, length 40
15:32:27.720527 IP 172.168.111.1 > 192.167.3.1: ICMP echo request, id 1, seq 143, length 40
15:32:27.720564 IP 192.167.3.1 > 172.168.111.1: ICMP echo reply, id 1, seq 143, length 40
15:32:28.722827 IP 172.168.111.1 > 192.167.3.1: ICMP echo request, id 1, seq 144, length 40
15:32:28.722867 IP 192.167.3.1 > 172.168.111.1: ICMP echo reply, id 1, seq 144, length 40
```
因为192.167.3.2通了，回程知道如何走，192.167.2.1自然也就通了
此时ping 192.167.2.2 自然不通，因为数据包到了net2，没有返程路由
net2执行，自然也就通了

```
ip net exec net2 ip route add 172.168.111.0/24 via 192.167.2.1 dev veth1
```
<font color=red>**net2主机访问window所有ip**</font>
主要在net1和net2中添加20.0/24路由策略即可
```
ip net exec net1 ip route add 192.168.20.0/24 via 192.167.3.2 dev veth2
ip net exec net2 ip route add 192.168.20.0/24 via 192.167.2.1 dev veth1 
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net2 ping 192.168.20.48                                    
PING 192.168.20.48 (192.168.20.48) 56(84) bytes of data.
64 比特，来自 192.168.20.48: icmp_seq=1 ttl=125 时间=0.295 毫秒
64 比特，来自 192.168.20.48: icmp_seq=2 ttl=125 时间=2.09 毫秒
```
<font color=red>**net1和net2可上网**</font>
指定默认的设备修改网关为window主机的网关。
>一定要保证网关地址是可以ping通后在添加默认网关路由
```
ip net exec net1 ip route add default dev veth2  #注意这里不能先指定网关，否则会报错
ip net exec net1 ip route change default via 192.168.20.1 
root@liaomin-Virtual-Machine:/home/liaomin# ping www.baidu.com
PING www.a.shifen.com (163.177.151.110) 56(84) bytes of data.
64 比特，来自 163.177.151.110 (163.177.151.110): icmp_seq=1 ttl=54 时间=8.90 毫秒
64 比特，来自 163.177.151.110 (163.177.151.110): icmp_seq=2 ttl=54 时间=9.02 毫秒
64 比特，来自 163.177.151.110 (163.177.151.110): icmp_seq=3 ttl=54 时间=14.0 毫秒
64 比特，来自 163.177.151.110 (163.177.151.110): icmp_seq=4 ttl=54 时间=9.24 毫秒
64 比特，来自 163.177.151.110 (163.177.151.110): icmp_seq=5 ttl=54 时间=9.14 毫秒
```
##### ipvlan
IPVlan 是从一个主机接口虚拟出多个虚拟网络接口。一个重要的区别就是所有的虚拟接口都有相同的 macv 地址，而拥有不同的 ip 地址。因为所有的虚拟接口要共享 mac 地址。
> ipvlan插件下，容器不能跟Host网络通信,

通过ip命令提供的网络隔离能力在同一个物理主机下创建两个虚拟网络net1和net2，创建两个网卡ipvlan1和ipvlan2，将ipvlan1绑定到net1指定ip为192.168.11.1/24，将ipvlan2绑定到net2指定ip为192.168.12.1/24,我这里宿主机的网卡eth0，ip为172.17.203.237
![在这里插入图片描述](/docs/images/content/devops/networking/basic/net_basic_02.md.images/fcda05b30902129264325f4e32ac9840.png)

```
cat <<EOF | bash
#存在添加两个网络net1和net2直接删除
((ip netns ls | grep net1 >/dev/null) && ip netns delete net1) && ((ip netns ls | grep net2 >/dev/null) && ip netns delete net2)
#新增网络net1和net2
ip netns add net1 && ip netns add net2
# 新增ipvlan1虚拟网卡和ipvalan2虚拟网卡挂载到物理网卡eth0是ipvlan类型，模式是l3
ip link add ipvlan1 link eth0 type ipvlan mode l3 && ip link add ipvlan2 link eth0 type ipvlan mode l3
#将ipvlan1加入到网络net1中，ipvlan2加入到网络net2中。
ip link set netns net1 ipvlan1 && ip link set netns net2 ipvlan2
#分表设置ipvlan1和ipvlan2的ip地址和网卡设备
ip netns exec  net1 ip addr add 192.168.11.1/24 dev ipvlan1 && ip netns exec  net2 ip addr add 192.168.12.1/24 dev ipvlan2
#启动lo和ipvlan网卡设备。
ip netns exec  net1 ip link set lo up && ip netns exec  net2 ip link set lo up
ip netns exec  net1 ip link set ipvlan1 up && ip netns exec  net2 ip link set ipvlan2 up
#分别设置各自网络环境的默认路由网卡设备是ipvlan1和ipvlan2，设置了该路由策略后，net1和net2互通（注意两个都需要设置）
ip netns exec net1 ip route add default dev ipvlan1 && ip netns exec net2 ip route add default dev ipvlan2
EOF
```
查看连个ipvlan的物理地址（发现ipvlan1和2物理地址都是00:15:5d:04:11:01和物理机物理地址相同）
```
root@liaomin-Virtual-Machine:/home/liaomin# ip link #查看物理主机网络
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT group default qlen 1000
   link/ether 00:15:5d:04:11:01 brd ff:ff:ff:ff:ff:ff 
root@liaomin-Virtual-Machine:/home/liaomin# ip netns exec net1 ip link #查看net1网络
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
32: ipvlan1@if2: <BROADCAST,MULTICAST,NOARP,UP,LOWER_UP> mtu 1500 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/ether 00:15:5d:04:11:01 brd ff:ff:ff:ff:ff:ff link-netnsid 0
root@liaomin-Virtual-Machine:/home/liaomin# ip netns exec net2 ip link  #查看net2网络
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
33: ipvlan2@if2: <BROADCAST,MULTICAST,NOARP,UP,LOWER_UP> mtu 1500 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/ether 00:15:5d:04:11:01 brd ff:ff:ff:ff:ff:ff link-netnsid 0
```
##### macvlan
每个网卡一个单独的mac地址，功能和ipvlan类似

```
# 创建两个 macvlan 子接口
ip link add link eth0 dev mac1 type macvlan mode bridge
ip link add link eth0 dev mac2 type macvlan mode bridge

# 创建两个 namespace
ip netns add net3
ip netns add net4

# 将两个子接口分别挂到两个 namespace 中
ip link set mac1 netns net3
ip link set mac2 netns net4

# 配置 IP 并启用
ip netns exec net3 ip a a 192.166.56.122/24 dev mac1
ip netns exec net3 ip l s mac1 up

ip netns exec net4 ip a a 192.166.56.123/24 dev mac2 #等价于ip add addr
ip netns exec net4 ip l s mac2 up   # ip l s等价于 ip link set 
```
可以查看到两个物理地址都不一样，和宿主机也不同

```
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net3 ip addr
1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
11: mac1@if2: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
    link/ether b6:e4:e5:e6:70:1f brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 192.166.56.122/24 scope global mac1
       valid_lft forever preferred_lft forever
    inet6 fe80::b4e4:e5ff:fee6:701f/64 scope link 
       valid_lft forever preferred_lft forever
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net4 ip addr 
1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
12: mac2@if2: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
    link/ether e2:f8:76:3e:36:16 brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 192.166.56.123/24 scope global mac2
       valid_lft forever preferred_lft forever
    inet6 fe80::e0f8:76ff:fe3e:3616/64 scope link 
       valid_lft forever preferred_lft forever
```
根据 macvlan 子接口之间的通信模式，macvlan 有四种网络模式：

1. private 模式:同一主接口下的子接口之间彼此隔离，不能通信。即使从外部的物理交换机导流，也会被无情地丢掉。
2. vepa(virtual ethernet port aggregator) 模式:这种模式下，子接口之间的通信流量需要导到外部支持 802.1Qbg/VPEA 功能的交换机上（可以是物理的或者虚拟的），经由外部交换机转发，再绕回来。
注：802.1Qbg/VPEA 功能简单说就是交换机要支持 发夹（hairpin） 功能，也就是数据包从一个接口上收上来之后还能再扔回去。
3. bridge 模式:模拟的是 Linux bridge 的功能，但比 bridge 要好的一点是每个接口的 MAC 地址是已知的，不用学习。所以，这种模式下，子接口之间就是直接可以通信的。
4. passthru 模式:只允许单个子接口连接主接口，且必须设置成混杂模式，一般用于子接口桥接和创建 VLAN 子接口的场景

#### 网桥
虽然 veth pair 可以实现两个 network namespace 之间的通信，但是当多个 namespace 需要通信的时候，就无能为力了。
讲到多个网络设备通信，我们首先想到的交换机和路由器。因为这里要考虑的只是同个网络，所以只用到交换机的功能。linux 当然也提供了虚拟交换机的功能，我们还是用 ip 命令来完成所有的操作。
>NOTE：和 bridge 有关的操作也可以使用命令 brctl，这个命令来自 bridge-utils 这个包，读者可以根据自己的发行版进行安装，使用方法请查阅 man 页面或者相关文档。

安装bridge-utils
```
apt install bridge-utils -y
```
查看所有网桥：
```
root@liaomin-Virtual-Machine:/home/liaomin# brctl show
bridge name     bridge id               STP enabled     interfaces
br0             8000.000000000000       no
```

首先我们来创建需要的 bridge，简单起见名字就叫做 br0。
```
[root@localhost ~]# ip link add br0 type bridge
[root@localhost ~]# ip link set dev br0 up
```
![在这里插入图片描述](/docs/images/content/devops/networking/basic/net_basic_02.md.images/c31fe960246bafb29d3e48cab1c32279.png)


创建两个网络net7和net8 分别创建veth0-1和veth2-3分表挂载网桥，实现不同网络间互通

```
ip net add net7 && ip net add net8
#新增veth10和veth11 设置ip和启动
ip link add veth10 type veth peer name veth11 && ip link set dev veth10 netns net7 && ip net exec net7 ip addr add 192.177.1.3/24 dev veth10 && ip net exec net7 ip link set dev veth10 up
# 将veth11添加到网桥
ip link set dev veth11 master br0 && ip link set dev veth11 up
#新增veth12和veth13 设置ip和启动
ip link add veth12 type veth peer name veth13 && ip link set dev veth12 netns net8 && ip net exec net8 ip addr add 192.177.1.4/24 dev veth12 && ip net exec net8 ip link set dev veth12 up
# 将veth11添加到网桥
ip link set dev veth13 master br0 && ip link set dev veth13 up
```
可以通过 bridge 命令（也是 iproute2 包自带的命令）来查看 bridge 管理的 link 信息：
```
root@liaomin-Virtual-Machine:/home/liaomin# bridge link
14: veth11@if15: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 master br0 state forwarding priority 32 cost 2 
16: veth13@if17: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 master br0 state forwarding priority 32 cost 2 
```
最后通过 ping 命令来测试网络的连通性：
```
root@liaomin-Virtual-Machine:/home/liaomin# ping 192.177.1.4
PING 192.177.1.4 (192.177.1.4) 56(84) bytes of data.
64 比特，来自 192.177.1.4: icmp_seq=1 ttl=49 时间=157 毫秒
64 比特，来自 192.177.1.4: icmp_seq=2 ttl=49 时间=158 毫秒
已发送 2 个包， 已接收 2 个包, 0% 包丢失, 耗时 1002 毫秒
rtt min/avg/max/mdev = 157.421/157.685/157.950/0.264 ms
root@liaomin-Virtual-Machine:/home/liaomin# ping 192.177.1.3
PING 192.177.1.3 (192.177.1.3) 56(84) bytes of data.
64 比特，来自 192.177.1.3: icmp_seq=1 ttl=49 时间=159 毫秒
64 比特，来自 192.177.1.3: icmp_seq=2 ttl=49 时间=158 毫秒
root@liaomin-Virtual-Machine:/home/liaomin# ip net exec net8 ping 192.177.1.3
PING 192.177.1.3 (192.177.1.3) 56(84) bytes of data.
64 比特，来自 192.177.1.3: icmp_seq=1 ttl=64 时间=0.055 毫秒
```
