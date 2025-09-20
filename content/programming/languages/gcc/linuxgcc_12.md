---
title: "linux下gcc编程12-window下clion编译调试redis7.0"
date: 2025-09-18T16:55:17+08:00
# bookComments: false
# bookSearchExclude: false
---

# redis介绍
Redis 通常被称为数据结构服务器。这意味着 Redis 通过一组命令提供对可变数据结构的访问，这些命令使用具有 TCP 套接字和简单协议的服务器-客户机模型发送。因此，不同的进程可以以共享的方式查询和修改相同的数据结构。

在 Redis 中实现的数据结构具有一些特殊属性:
1. 注意将它们存储在磁盘上，即使它们总是被提供并修改到服务器内存中。这意味着 Redis 是快速的，但它也是非易失性的。
2. 数据结构的实现强调了内存效率，因此与使用高级语言建模的相同数据结构相比，Redis 内部的数据结构可能会使用更少的内存
3. 提供了一些在数据库中自然可以找到的特性，比如复制、可调的持久性级别、集群和高可用性。

另一个很好的例子是将 Redis 视为 memcached 的一个更复杂的版本，其中的操作不仅仅是 SET 和 GET，还包括处理复杂数据类型(如 List、 Set、有序数据结构等)的操作。

如果你想知道更多，这是一个选定的起点列表:
- Redis 数据类型简介  https://redis.io/topics/data-types-intro
- 直接在浏览器中尝试 Redis https://try.redis.io
- Redis 命令的完整列表 https://redis.io/commands
- 官方 Redis 文档中还有更多内容 https://redis.io/documentation

# window下编译redis
window编译redis的目的主要是用于在clion下调试redis，方便代码阅读。
安装cygwin环境：参考:[nginx编译教程中安装cygwin章节](https://blog.csdn.net/liaomin416100569/article/details/105127557?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167109847316800182779069%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=167109847316800182779069&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-105127557-null-null.nonecase&utm_term=nginx&spm=1018.2226.3001.4450)

下载redis7.0代码：

```
git clone -b 7.0 https://github.com/redis/redis.git
```

打开Cygwin64 Terminal，执行命令：make。

```
make
cd src && make all
make[1]: Entering directory '/cygdrive/d/code1/redis-7.0/src'
    CC Makefile.dep
```
编译过程中会报错
```
debug.c:1759:5: error: unknown type name ‘Dl_info’
```
查看src/debug.c有代码如下：
```
#include <dlfcn.h>
```

找到cygwin安装目录下的usr\include\dlfcn.h,有一段代码

```
#if __GNU_VISIBLE
typedef struct Dl_info Dl_info;

struct Dl_info
{
   char        dli_fname[PATH_MAX];  /* Filename of defining object */
   void       *dli_fbase;            /* Load address of that object */
   const char *dli_sname;            /* Name of nearest lower symbol */
   void       *dli_saddr;            /* Exact value of nearest symbol */
};

extern int dladdr (const void *addr, Dl_info *info);
#endif
```
明显是判断了__GNU_VISIBLE这个宏，可以copy一个dlfcn1.h
在#if __GNU_VISIBLE前加上一个#define __GNU_VISIBLE 1

```
#define __GNU_VISIBLE 1
#if __GNU_VISIBLE
typedef struct Dl_info Dl_info;

struct Dl_info
{
   char        dli_fname[PATH_MAX];  /* Filename of defining object */
   void       *dli_fbase;            /* Load address of that object */
   const char *dli_sname;            /* Name of nearest lower symbol */
   void       *dli_saddr;            /* Exact value of nearest symbol */
};

extern int dladdr (const void *addr, Dl_info *info);
#endif
```

修改redis源代码src/debug.c修改为：
```
#include <dlfcn1.h>
```
重新执行make，src下生成了redis-开头的exe文件
![在这里插入图片描述](/docs/images/content/programming/languages/gcc/linuxgcc_12.md.images/d147910ea565f84aee81517422f70797.png)
从cygwin目录/bin目录拷贝cygwin1.dll到redis的src源码目录，同时将redis源码根目录的redis.conf拷贝到src源码目录。
尝试双击redis-server.exe
![在这里插入图片描述](/docs/images/content/programming/languages/gcc/linuxgcc_12.md.images/308c0fb77c4b0ca25bc9c4f1e8791206.png)
开启redis-cli连接
![在这里插入图片描述](/docs/images/content/programming/languages/gcc/linuxgcc_12.md.images/1c60c78f0e4a66c2602ab4902af061ba.png)
# clion调试redis
>注意clion在2021年后的版本才支持makefile的程序调试，需要升级到版本>=clion2021
>同时需要先用上面的命令编译出exe文件。
clion中确保插件Makefile是安装的
![在这里插入图片描述](/docs/images/content/programming/languages/gcc/linuxgcc_12.md.images/9972e64357b3004b2df822c72ca30598.png)
clion打开redis项目
设置下cygwin环境
![在这里插入图片描述](/docs/images/content/programming/languages/gcc/linuxgcc_12.md.images/6b166e23023a248d165aa2ff08fe4bad.png)
指定make文件
![在这里插入图片描述](/docs/images/content/programming/languages/gcc/linuxgcc_12.md.images/ac5af9b39aab9c8ff941a56382e4b1d9.png)

創建 Custom Build Targets，这步的意思是构建和清除指定的命令
新建一个
![在这里插入图片描述](/docs/images/content/programming/languages/gcc/linuxgcc_12.md.images/65d86dbfe77d9f5101b33b8022560f7d.png)
toolchain选择cygwin
build 右侧...新建一个build
![在这里插入图片描述](/docs/images/content/programming/languages/gcc/linuxgcc_12.md.images/b24e1603f064467e3eb968aa9e4abe20.png)
在新建一个clean
![在这里插入图片描述](/docs/images/content/programming/languages/gcc/linuxgcc_12.md.images/6c31f01d03c611c889ede0959e385234.png)
Run/Debug Configuration新增一个配置
指定Custom Build Targets和可执行文件
![在这里插入图片描述](/docs/images/content/programming/languages/gcc/linuxgcc_12.md.images/9ffe05a787c003209766ae4d4071911e.png)
src/server.c 中的main方法打个断点，debug启动
![在这里插入图片描述](/docs/images/content/programming/languages/gcc/linuxgcc_12.md.images/f4b643723d5db9bcca18df7805281d05.png)
>src/Makefile文件%.o: %.c .make-prerequisites任务用于编译，如果需要加gcc参数可加载这里
# redis源码分析
调试环境已经备好，后续有空在慢慢分析。