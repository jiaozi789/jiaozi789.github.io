---
title: "android逆向攻防01-http抓包"
date: 2025-09-18T16:55:17+08:00
weight: 1
# bookComments: false
# bookSearchExclude: false
---

**一.计算机网络分类**

   覆盖范围分为：

     局域网 LAN（：Local Area Network） 高数据传输  低延迟 低误码率 组件成本低

     城域网 MAN  （MetroPolitan Area Network） 主要使用光纤传输

     [广域网](https://so.csdn.net/so/search?q=%E5%B9%BF%E5%9F%9F%E7%BD%91&spm=1001.2101.3001.7020) WAN (Wide Area Network)                      主要使用光纤传输

     互联网 (Network)

**二.计算机网络拓扑结构**

     用于描述计算机 网线以及其他网络设备的配置方式 是网络物理布局的一种方式

1.总线型拓扑   所有的计算机通过一条总线来进行连接 如果一台机器发送数据其他的机器就必须先暂停发送等待 发送效率较低

![](/docs/images/content/devops/networking/basic/net_basic.md.images/image_1.png)  

2.星型拓扑  目前最流行网络拓扑  主要是使用一个中心的集线器或者交换机来进行管理 所以的pc连接该中心 发送和接受数据 每台pc可以单独发送接受数据不受其他pc机器影响

![](/docs/images/content/devops/networking/basic/net_basic.md.images/image_2.png)  

常见的公司星型网络架构  不同网络之间形成了一个树形网络架构  
不同公司内部搭建局域网 通过isp运营商接入外部城域网将多个城市网络接入广域网

  ![](/docs/images/content/devops/networking/basic/net_basic.md.images/image_3.png)

3.环型拓扑  pc机器之间使用环形结构通信 比如 pc1 发送数据给pc3 必须经过pc2 一般不使用在局域网 

![](/docs/images/content/devops/networking/basic/net_basic.md.images/image_4.png)  

4.网状拓扑

    每台pc机器之间都进行连接  比如 Apc和B,C,D都有连接 缺点是 需要的线路特别多

![](/docs/images/content/devops/networking/basic/net_basic.md.images/image_5.png)  

**三.网络互连设备**  

物理层 ：中继器（Repeater，也叫放大器），集线器。  
  
数据链路层 ：网桥，交换机。  
  
网络层 ：路由器。  
  
网关 ：网络层以上的设备。  

设备解释：

网卡 NIC(NetWork Interface Card) 计算机和计算机进行通信的设备 每台计算机最少配置一个网卡

中继器 是信号放大器 信号在传输过程中可能会丢失数据 中继器可以重新恢复数据

集线器 都是星型拓扑结构的中心处理器 OSI7层物理层 传输的是bit 所有的接口争用带宽

交换机  也可以是星型拓扑结构的中心处理器 OSI7层的数据链路层 传输的是数据帧 包含了发送和接受方的物理地址等信息

   局域网一般都使用交换机 因为交换器网络接口多 接口之间处理的流量大 接口之间独立拥有带宽

网桥一般是两个不同网络间进行数据交换的桥梁 网桥会记录需要桥接的两个网络中的所有网络成员的物理地址 

网关  协议转换器 用于连接差别较大 协议不同的网络之间连接 可以打包数据重新转换成其他网络格式的数据包并发送 作用 iso7层

路由器是一种特殊的网关 iso3层 比如计算机 网卡上都需要设置网关上网  

     一般路由器都有两个以上的网卡 

          路由器网卡1 ip地址为 192.168.1.1  其他连接该路由器的局域网主机都会分配一个 192.168.1网段的IP地址  局域网主机的网关配置必须为路由器的ip 1                      192.168.1.1

          路由器 网卡2  用于连接互联网  连接后会存在一个公网ip   

      当局域网主机发起一个外网访问的ip包后 包会被网关192.168.1.1路由器接受 接受后 重新打包数据（修改源ip为外网ip） 发送给网卡2  网卡2访问外网结果后获取结果后 重新打包数据 发送给网卡1  原路返回响应给局域网主机

      centos主机 可以通过修改 echo 1 > /proc/sys/net/ipv4/ip\_forward  启用路由功能

  

**三.OSI七层模型**  

![](http://hi.csdn.net/attachment/201202/10/0_1328873801dE3r.gif)  

每层具体用途

![](/docs/images/content/devops/networking/basic/net_basic.md.images/image_7.png)  

具体描述：

![](/docs/images/content/devops/networking/basic/net_basic.md.images/image_8.png)  

![](/docs/images/content/devops/networking/basic/net_basic.md.images/image_9.png)  

![](/docs/images/content/devops/networking/basic/net_basic.md.images/image_10.png)  

![](/docs/images/content/devops/networking/basic/net_basic.md.images/image_11.png)  

![](/docs/images/content/devops/networking/basic/net_basic.md.images/image_12.png)