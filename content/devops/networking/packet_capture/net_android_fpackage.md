---
title: "android逆向攻防01-http抓包"
date: 2025-09-18T16:55:17+08:00
weight: 1
# bookComments: false
# bookSearchExclude: false
---

# 概述
网络抓包，是Android应用逆向分析的重中之重，很多时候我们拿到一个APP，不知道从何入手分析，往往是从抓包开始，先弄清楚他与服务器通信的内容，如果一目了然，我们完全可以照搬，自行写一个程序来模拟，如果有一些加密字段和随机字段，也不用担心，我们可以从抓包中了解到一些关键的URL和session之类的信息，然后再反编译分析代码的时候，这些字符串可以帮助我们更快的定位关键代码所在之处。

android抓包的方式有以下几种：
1. 基于代理的https根证书替换抓包。
2. android系统安装抓包软件。
3. 使用sslhock抓包pcap文件分析。

> 该内容仅供用于学习目的，请勿用于商业目的。
# 抓包实战
## 代理抓包
使用代理抓包的工具非常多，比如burpsuite，fiddler，charles等。
其中burpsuite抓包之前写过，地址：https://blog.csdn.net/liaomin416100569/article/details/129176916
无论是fidder和charles都是充当了一个中间人代理的角色来对HTTPS进行抓包:
- 截获客户端向发起的HTTPS请求，佯装客户端，向真实的服务器发起请求。
- 截获真实服务器的返回，佯装真实服务器，向客户端发送数据。
- 获取了用来加密服务器公钥的非对称秘钥和用来加密数据的对称秘钥。
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/a22beb2cc2fc195618fa0bce052734e5.png)
这里演示charles，模拟器使用夜神模拟器。
> 使用fiddler导出根证书，使用下面相同的方法，在chrome中就无法抓包，不知道是版本的还是其他问题，charles和burpsuite正常。
### charles安装
建议安装使用最新版，官方下载地址 https://www.charlesproxy.com/download
这是我的注册序列号
Registered Name:	jiaozi
License Key:	5363faa4184fb6fbcb

这是免费共享序列号提供站：https://www.zzzmode.com/mytools/charles/。
github开源地址:https://github.com/8enet/Charles-Crack
### 浏览器抓包
点击proxy-proxy settings 设置代理端口，比如我设置的8881
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/87f37b09f15e822f6d62ab71c2b0923d.png)
点击 proxy-ssl proxy settings ，新增一个include *:*
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/caa35cc917b35c080d3f411f927bf17e.png)
点击help - ssl proxying - Install Charles Root Certificate，证书弹出后点击安装证书-选择本地计算机
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/3c8e429f59db9f080f63bb866326d38e.png)
点击浏览选择：受信任的根证书颁发机构
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/bb881313b34fcfcd5157dfa650198552.png)
在浏览器上通过SwitchyOmega添加代理绑定到 ip:8881,浏览器切换到该场景，抓包
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/d9ad9b65d576dfa869d463dcd64729d3.png)
### 手机抓包
点击charles help-ssl proxying   Save Charles Root Certificate...
选择pem格式保存
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/9b2346ffdd7a05cab0fc0c8754806ab2.png)

因为android模拟器内部存储证书的名字必须是pem的hash值.0方式存储，所以需要知道他的hash值
burpsuite生成的根证书hash值是 9a5ba575，所以可以直接将cacert.pem重命名为:9a5ba575.0
当然也可以用openssl确认下
```
openssl x509 -subject_hash_old -in cacert.pem

C:\Users\liaomin>openssl x509 -subject_hash_old -in d:\test\cert\chares.pem
e4a84eb5
-----BEGIN CERTIFICATE-----
MIIFRjCCBC6gAwIBAgIGAYbkTnV4MA0GCSqGSIb3DQEBCwUAMIGnMTgwNgYDVQQD
DC9DaGFybGVzIFByb3h5IENBICgxNSBNYXIgMjAyMywgREVTS1RPUC1GU0VERTNQ
KTElMCMGA1UECwwcaHR0cHM6Ly9jaGFybGVzcHJveHkuY29tL3NzbDERMA8GA1UE
CgwIWEs3MiBMdGQxETAPBgNVBAcMCEF1Y2tsYW5kMREwDwYDVQQIDAhBdWNrbGFu
ZDELMAkGA1UEBhMCTlowHhcNMjMwMzE0MDgwNjMyWhcNMjQwMzEzMDgwNjMyWjCB
pzE4MDYGA1UEAwwvQ2hhcmxlcyBQcm94eSBDQSAoMTUgTWFyIDIwMjMsIERFU0tU
T1AtRlNFREUzUCkxJTAjBgNVBAsMHGh0dHBzOi8vY2hhcmxlc3Byb3h5LmNvbS9z
c2wxETAPBgNVBAoMCFhLNzIgTHRkMREwDwYDVQQHDAhBdWNrbGFuZDERMA8GA1UE
CAwIQXVja2xhbmQxCzAJBgNVBAYTAk5aMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A
MIIBCgKCAQEAksPed8nM3xcFBapr93Pqso1vTJp8Dl5tKu831Oxx0jCeOFc8TvWn
Bp6/4UFsfxqj58q0oR6EzJ9wRw1AuABHGIHFn0YsRmZzKudr8W5N0iRoMLz5OE0j
ycN5PIZHJ2f1R6+V82JHOzFHJU/zV93Ap6870PO6Sgutjk0tqnPfs0o/5kyHkle7
JLgG/HjTRA7jaKWUXpqAzgb64hzcEM3D0GdvxDZ4DHGlShB7WndsH4cqW7hE72Jz
zo8UW9lXRACpEjtfhPTVk8KWDijQflthkOpUq5jLo75QlP02j4YRxPJ9st9w2XIF
G9E63MSzY9k1paEwoUY65QVQ5HRQrVtLHwIDAQABo4IBdDCCAXAwDwYDVR0TAQH/
BAUwAwEB/zCCASwGCWCGSAGG+EIBDQSCAR0TggEZVGhpcyBSb290IGNlcnRpZmlj
YXRlIHdhcyBnZW5lcmF0ZWQgYnkgQ2hhcmxlcyBQcm94eSBmb3IgU1NMIFByb3h5
aW5nLiBJZiB0aGlzIGNlcnRpZmljYXRlIGlzIHBhcnQgb2YgYSBjZXJ0aWZpY2F0
ZSBjaGFpbiwgdGhpcyBtZWFucyB0aGF0IHlvdSdyZSBicm93c2luZyB0aHJvdWdo
IENoYXJsZXMgUHJveHkgd2l0aCBTU0wgUHJveHlpbmcgZW5hYmxlZCBmb3IgdGhp
cyB3ZWJzaXRlLiBQbGVhc2Ugc2VlIGh0dHA6Ly9jaGFybGVzcHJveHkuY29tL3Nz
bCBmb3IgbW9yZSBpbmZvcm1hdGlvbi4wDgYDVR0PAQH/BAQDAgIEMB0GA1UdDgQW
BBQdxz00EUr7sROKkc3amn5njk1GwTANBgkqhkiG9w0BAQsFAAOCAQEAa+JJpXin
oeWzDqfVcn7N6nxXDSvicCaDZx/lXXIxvRrmR4Wbq6q6s6Jeft8WxKroPp91LiL1
U/Wd48y5fqwMlgMxqcrkeblWzz9AjUj0A6NCfOeOSrAqZ0Ph9R0mPQag/nM/2pez
76tHmBifK8ZiYOZqvU9ui8jrWghdY2RIo9Mm8jybEyahuX4Vs18nGLxxYJ+q4+l/
IZSSxOdUcAQilAW2ek0M/IVVIxQe1wLvl5FTDMnuFXm0JYjXB6gmnVe6Hiclv8kS
igrOUzyZcgxkYqgYlSEb1yn1WxPB7ccwv43jDC3Hx/1oX46f07DxJg3+50ZKzR1Y
EV63aZEaoP67wA==
-----END CERTIFICATE-----

```
pem存储为：e4a84eb5.0

上传证书到模拟器
打开文件资源管理器，进入夜神模拟器的安装目录，找到nox_adb.exe或者adb.exe程序，将他的路径加入到环境变量Path中，或者cmd直接到夜神目录执行命令
查看模拟器设备

```
D:\Program Files\Nox\bin>adb devices
List of devices attached
127.0.0.1:62001 device
```

输入nox_adb.exe connect 127.0.0.1:62001即可以连接到adb，或者是adb connect 127.0.0.1:62001

然后依次执行以下命令，在查看系统证书就会发现成功安装。
```
adb root // 提升到root权限
adb remount // 重新挂载system分区
adb push D:\test\cert\e4a84eb5.0 /system/etc/security/cacerts/ //将证书放到系统证书目录
```
模拟器设置代理，参考：https://blog.csdn.net/liaomin416100569/article/details/129176916
浏览器访问https站点，查看证书是否是charles颁发的证书
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/5024c00dd28dcb8ee0042bdd9ec15313.png)
打开其他apps，测试抓包成功

## 安装app抓包
HttpCanary黄鸟抓包工具是一款专为Android用户设计的手机软件，会实时监控手机，防止恶意软件篡改手机系统。黄鸟对于新手和老手来说都是一个很好的工具，可以轻松流畅的抓取网页的HTTP/HTTPS数据，让用户更方便的分析当前网页，方便性很高，而且数据全面，显示直观清晰，实用性大大提高。

安装MT管理器  https://mt2.cn/

在 MT 管理器中进入路径 /data/data/com.guoshi.httpcanary/cache/ 的目录下将 HttpCanary.pem 证书文件复制一份，并将文件名修改成 87bc3517.0，如果 HttpCanary.pem 证书文件不存在的话，打开 HttpCanary 软件，在设置里面尝试安装根证书已生成证书文件，然后点击导出证书。
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/a44c094a9a6464983b355db84efa46f3.png)
在 MT 管理器中将刚才复制出来并修改了文件名的证书文件 87bc3517.0 移动到 /system/etc/security/cacerts/ 目录下
如果是导出的.0证书，可以位于：HttpCanary/cert目录里
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/46bb2b65badf43733585626ff1fdbdc3.png)
点击.0文件移动到右侧系统证书目录
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/dbb89d04d7bea8cdd7008fe6b764efd7.png)
并通过 MT 管理器修改文件权限（长按文件 — 属性 — 权限）为 644
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/627d8dc1ab1689a031cb2977ab27020b.png)
在 /data/data/com.guoshi.httpcanary/cache/ 的目录下新建一个空文件，文件名为 HttpCanary.jks，并通过 MT 管理器修改文件权限（长按文件 — 属性 — 权限）为 660 即所有者读写，其他无权限，目的是让httpcanary认为已经安装了证书就不会每次启动都弹出提示需要安装证书。

至此就成功安装根证书了，可以打开 HttpCanary 在设置 — HttpCanary 根证书 — 卸载 HttpCanary 根证书 — 系统，在系统这一栏中检查是否有 HttpCanary 的字样，有就代表根证书安装成功。

抓包效果

![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/1fcc1bd00aa6a0d05ce2d42bdef6b705.png)

## sslhock抓包
r0capture安卓应用层抓包通杀脚本,利用frida hookssl api抓取数据包，导出为pcap文件使用wireshark分析。
- 仅限安卓平台，测试安卓7、8、9、10、11、12 可用 ；
- 无视所有证书校验或绑定，不用考虑任何证书的事情；
- 通杀TCP/IP四层模型中的应用层中的全部协议；
- 通杀协议包括：Http,WebSocket,Ftp,Xmpp,Imap,Smtp,Protobuf等等、以及它们的SSL版本；
- 通杀所有应用层框架，包括HttpUrlConnection、Okhttp1/3/4、Retrofit/Volley等等；
- 无视加固，不管是整体壳还是二代壳或VMP，不用考虑加固的事情；
- 如果有抓不到的情况欢迎提issue，或者直接加vx：r0ysue，进行反馈~

### 安装frida
安装conda
在conda中添加一个虚拟环境  python3.7 [参考](https://blog.csdn.net/liaomin416100569/article/details/83745320?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167902009816800226557992%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=167902009816800226557992&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-83745320-null-null.blog_rank_default&utm_term=conda&spm=1018.2226.3001.4450)

激活你的环境
```
C:\Users\liaomin>activate r0capture

(r0capture) C:\Users\liaomin>
```
安装frida-tool,会自动安装对应版本的frida

```
pip install install frida-tools
```
当然环境下查看frida版本

```
查看frida版本frida --version
(r0capture) C:\Users\liaomin>frida --version
16.0.11
```
查看夜神模拟器 cpu版本
```
(r0capture) C:\Users\liaomin>adb shell
127|z3q:/ # getprop ro.product.cpu.abi
x86
```
根据cpu版本去下载相应[frida-server](https://github.com/frida/frida/releases),手机是x86的，找到相应的服务器server,如下
[frida-server-16.0.11-android-x86.xz](https://github.com/frida/frida/releases/download/16.0.11/frida-server-16.0.11-android-x86.xz)
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/b6f8795f05003cb2d61c282eb46a307e.png)
将frida-server下载下来，加压出来，为了简单，重命名为frida-server，在此目录打开cmd 运行下面的命令
window执行
```
adb push frida-server /data/local/tmp
adb forward tcp:27042 tcp:27042      # 注意如果模拟器不开启转发会导致frida-ps -R 就会执行报错(r0capture也无法连接)，frida-ps -U 可执行
adb forward tcp:27043 tcp:27043       # 注意如果模拟器不开启转发会导致frida-ps -R 就会执行报错，frida-ps -U 可执行
```
>注意如果模拟器不开启转发会导致frida-ps -R 就会执行报错(r0capture也无法连接)，frida-ps -U 可执行
>android模拟器重启后需要重新执行adb forward和启动下面的./frida-server

模拟器android执行

```
adb shell
  cd /data/local/tmp
  chmod 755 ./frida-server
  nohup ./frida-server &
```
启动frida-server后，进入之前python3.7的r0capture虚拟环境下执行
frida-ps -U   和  frida-ps -R  都能够抓取到android进程列表即可
```
(r0capture) C:\Users\liaomin> frida-ps -R
 PID  Name
----  ------------------------------------------
3230  MT管理器
1805  adbd
2465  android.ext.services
2497  android.process.acore
2585  android.process.media
1883  audioserver
2518  cameraserver
2730  com.android.carrierconfig
2912  com.android.inputmethod.pinyin
2549  com.android.launcher3
2717  com.android.managedprovisioning
2773  com.android.onetimeinitializer
2309  com.android.phone
2534  com.android.printspooler
2760  com.android.providers.calendar
2253  com.android.systemui
```

### 安装r0capture
安装前置依赖库

```
pip install loguru
pip install clickx
pip install hexdump
```
下载r0capture脚本

```
https://github.com/r0ysue/r0capture
主要是以下三个文件，缺一不可
r0capture.py
script.js
myhexdump.py
```
在目录下执行

```
python r0capture.py -U -f com.xhnf.piano -v -p 2.pcap
```
>app的包名可以使用androidkiller打开apk查看

确定不再抓包 ctrl+c退出即可 生成的2.pcap直接用wireshark打开
![在这里插入图片描述](/docs/images/content/devops/networking/packet_capture/net_android_fpackage.md.images/42904aea03a93fa7fcc72399ff776270.png)
在http的包商右键-追踪流-http流查看请求响应报文。