---
title: "linux下gcc编程07-使用linux下c库函数"
date: 2025-09-18T16:55:17+08:00
# bookComments: false
# bookSearchExclude: false
---

**一 。linux c库函数简介**

 
linux下用于c编程的头文件 位于以下目录

 
```
/usr/local/include
/usr/lib/gcc-lib/target/version/include
/usr/target/include
/usr/include
```

 
库文件一般位于

 
```
/usr/lib或/lib或/lib64
```

 
**二。常用库函数分类示例**

 
1.文件操作库

 
  参考linuxc常用c函数手册 文件操作篇（[http://net.pku.edu.cn/~yhf/linux_c/](http://net.pku.edu.cn/~yhf/linux_c/)）

 
```
#include <stdio.h> //包含io流的库 标准的输入和输出
#include "string.h"
/**
 * fcntl.h，是unix标准中通用的头文件，其中包含的相关函数有 open，fcntl，shutdown，unlink，fclose等！
 * 就是定义了文件操作的常量
 */
#include <fcntl.h>
#include "stdlib.h"
/**
 * unistd.h 是 C 和 C++ 程序设计语言中提供对 POSIX 操作系统 API 的访问功能的头文件的名称
 * unistd.h 中所定义的接口通常都是大量针对系统调用的封装（英语：wrapper functions），如 fork、pipe 以及各种 I/O 原语（read、write、close getpid() 等等）
 */
#include "unistd.h"
/**
 * 是Unix/Linux系统的基本系统数据类型的头文件，含有size_t，time_t，pid_t等类型。
 */
#include "sys/types.h"
 
#define READ_SIZE 140
int main() {
 
    /*-------获取当前的进程id*/
    pid_t pt=getpid();//pid_t在sys/types.h中 getpid()在unistd.h中
    printf("当前程序的进程id：%d\n",pt);
    /*-------打开一个文件描述符*/
    int fileDescp=open("c:/a1.txt",O_RDONLY);//open函数unistd中 常量O_RDONLY定义在fcntl.h
    char s[READ_SIZE];
    memset(s,'\0',READ_SIZE);//将所有的字节初始化 \0
    /*-------读取指定长度字节*/
    //注意项目设置的什么字符集 读取文件时 如果文件不是该字符集就会读出乱码
    read(fileDescp,s,READ_SIZE);
    printf("读取字符串:%s\n",s);
    /*-------读取全部数据*/
    lseek(fileDescp,SEEK_SET,0);//设置从0开始读取
    memset(s,'\0',READ_SIZE);
    while((read(fileDescp,s,READ_SIZE))>0){
        printf("读取到的数据 :%s\n",s);
    }
    /*-------关闭文件描述符*/
    close(fileDescp);
    /*-------读取全部数据*/
    //创建一个文件指定权限 权限数字参考 http://www.runoob.com/linux/linux-file-attr-permission.html
    int fd1 = creat("c:/test.log",777);
    write(fd1, s, strlen(s));
    close(fd1);
    //getchar();
    //return 0;
}
```

 
2.进程操作库

 
》》父子进程
  

 
```
#include  <stdio.h>
#include <stdlib.h>
#include "string.h"
#include <sys/unistd.h>
#include <sys/wait.h>
void childProcessTest();
int main() {
    childProcessTest();
}
 
/**
 * 演示子进程例子
 */
void childProcessTest(){
    //------------获取进程id
    pid_t pid=getpid();
    printf("获取当前进程id:%d\n",pid);
 
    int i=100;
    pid_t cpid;
    //------------fork()会创建一个子进程 父子进程都会调用fork 父进程返回子进程的id 子进程=0
    //            注意如果父进程先运行完 子进程也会自动停止 不会执行完成代码
    //            进程子进程之前的变量都会生成一个拷贝 不能共享
    /**
     * fork（）与vfock（）都是创建一个子进程，那他们有什么区别呢？总结有以下三点区别：
        1.  fork  （）：子进程拷贝父进程的数据段，代码段
            vfork （ ）：子进程与父进程共享数据段
        2.  fork （）父子进程的执行次序不确定
            vfork 保证子进程先运行，在调用exec 或exit 之前与父进程数据是共享的,在它调用exec
             或exit 之后父进程才可能被调度运行。
     */
    if((cpid=fork())>0){
        //父进程执行
        i++;
        printf("父进程i=:%d\n",i);
        int status;
        //------------wait（）函数在wait.h中 等待子进程退出 一直阻塞 status可以获取到子进程传递状态数字 通过WEXITSTATUS宏转换才能获取到11
        //           waitpid(cpid,&status,0); waitpid假设有多个子进程的情况下可以等待某个指定pid的子进程
        pid_t cc=wait(&status);
        printf("父进程获取到子进程%d的退出状态码:%d\n",cc,WEXITSTATUS(status));
    }else{
        //子进程执行
        //------------getppid()取得父进程的进程识别码
        printf("子进程id:%d,子进程获取父进程id:%d\n",getpid(),getppid());
        //------------sleep()让进程暂停执行一段时间 单位秒
        sleep(3);
        i++;
        printf("子进程i=:%d\n",i);
        //------------退出当前进程将参数11返回给父线程 并且会传递SIGCHLD信号给父进程，父进程可以由wait函数取得子进程结束状态。
        exit(11);
    }
}
```

 
》》system和exec族函数区别
  

 
```
#include  <stdio.h>
#include <stdlib.h>
#include "string.h"
#include <sys/unistd.h>
#include <sys/wait.h>
 
int main() {
    //system()会调用fork()产生子进程，由子进程来调用
    system("dir c:/");
 
    //----------execl将当前execl后面所有代码替换成 execl执行 内容 后面代码不执行 第一个参数必须全路径 参数2表示文件名 参数三才表示第一个参数
    //execl("C:\\Windows\\System32\\notepad","notepad","c:/a.txt",NULL);
    //----------从PATH 环境变量中查找文件并执行 第一个路径不加全路径
    //execlp("notepad","notepad","c:/a.txt",NULL);
    //----------和exel一模一样 不从path中找 只有两个参数 第二个参数定义在一个数组中
    char * args[]={"notepad","c:/a.txt",NULL};
    execv("C:\\Windows\\System32\\notepad",args);
    printf("测试是否打印");
 
}
```

 
》》进程IPC（进程通信）

 
linux常用的进程间的通讯方式

 
（1）、管道(pipe)：管道可用于具有亲缘关系的进程间的通信，是一种半双工的方式，数据只能单向流动，允许一个进程和另一个与它有共同祖先的进程之间进行通信。

 
（2）、命名管道(named pipe)：命名管道克服了管道没有名字的限制，同时除了具有管道的功能外（也是半双工），它还允许无亲缘关系进程间的通信。命名管道在文件系统中有对应的文件名。命名管道通过命令mkfifo或系统调用mkfifo来创建。

 
（3）、信号（signal）：信号是比较复杂的通信方式，用于通知接收进程有某种事件发生了，除了进程间通信外，进程还可以发送信号给进程本身；linux除了支持Unix早期信号语义函数sigal外，还支持语义符合Posix.1标准的信号函数sigaction（实际上，该函数是基于BSD的，BSD为了实现可靠信号机制，又能够统一对外接口，用sigaction函数重新实现了signal函数）。

 
（4）、消息队列：消息队列是消息的链接表，包括Posix消息队列system V消息队列。有足够权限的进程可以向队列中添加消息，被赋予读权限的进程则可以读走队列中的消息。消息队列克服了信号承载信息量少，管道只能承载无格式字节流以及缓冲区大小受限等缺

 
（5）、共享内存：使得多个进程可以访问同一块内存空间，是最快的可用IPC形式。是针对其他通信机制运行效率较低而设计的。往往与其它通信机制，如信号量结合使用，来达到进程间的同步及互斥。

 
（6）、内存映射：内存映射允许任何多个进程间通信，每一个使用该机制的进程通过把一个共享的文件映射到自己的进程地址空间来实现它。

 
（7）、信号量（semaphore）：主要作为进程间以及同一进程不同线程之间的同步手段。

 
（8）、套接字（Socket）：更为一般的进程间通信机制，可用于不同机器之间的进程间通信。起初是由Unix系统的BSD分支开发出来的，但现在一般可以移植到其它类Unix系统上：Linux和System V的变种都支持套接字。

 
代码写了前5个 

 
```
#include  <stdio.h>
#include <stdlib.h>
#include "string.h"
#include <sys/unistd.h>
#include <sys/wait.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
 
#include <sys/types.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#define BSIZE 200
#define FIFO "c:/alink"
 
 
/**
 * (1)无名管道
 *  用于具有亲缘关系父子的进程间的通信，是一种半双工的方式，数据只能单向流动，允许一个进程和另一个与它有共同祖先的进程之间进行通信
 *   pipe (int __fildes[2]); 产生管道 数组0读管道， 数组1表示写管道
 *   这里模拟父进程给子进程写入一个hello my son
 */
void pipeTest(){
 
    int __fildes[2];
    pipe(__fildes);
    pid_t  pid;
    char buffer[BSIZE];
    //父子进程都会有读写管道的拷贝
    if((pid=fork())>0){//父进程写 关闭读的管道
        close(__fildes[0]);
        memset(buffer,'\0',BSIZE);
        char* writeChar="hello my 儿子";
        memcpy(buffer,writeChar, strlen(writeChar));
        write(__fildes[1],buffer,BSIZE);
        close(__fildes[1]);
        wait(NULL);
 
    }else{//子进程读 关闭写的管道
        sleep(2);
        close(__fildes[1]);
        read(__fildes[0],buffer,BSIZE);
        printf("子进程读取到数据:%s",buffer);
        close(__fildes[0]);
 
    }
}
/**
 * （2）管道克服了管道没有名字的限制，同时除了具有管道的功能外（也是半双工），它还允许无亲缘关系进程间的通信。
 * 命名管道在文件系统中有对应的文件名。命名管道通过命令mkfifo或系统调用mkfifo来创建。
 * 这里模拟父进程给子进程写入一个hello my son
 * @return
 */
void namePipeTest(){
    //-------------unlink()会删除参数pathname指定的文件。如果该文件名为最后连接点，但有其他进程打开了此文件，则在所有关于此文件的文件描述词皆关闭后才会删除。如果参数pathname为一符号连接，则此连接会被删除。
    //返回值 成功则返回0，失败返回-1，错误原因存于errno
    unlink(FIFO);
    if(errno==EROFS){
        printf("文件存在于只读文件系统内");
    }
    //--------------mkfifo注意在 <sys/stat.h>头文件中
    /*
    kfifo()会依参数pathname建立特殊的FIFO文件，该文件必须不存在，而参数mode为该文件的权限（mode%~umask），因此umask值也会影响到FIFO文件的权限。Mkfifo()建立的FIFO文件其他进程都可以用读写一般文件的方式存取。当使用open()来打开FIFO文件时，O_NONBLOCK旗标会有影响
    fifo类似于队列 阻塞队列 没有数据读 默认阻塞等到有数据写入 O_NONBLOCK表示不阻塞
     1、当使用O_NONBLOCK 旗标时，打开FIFO 文件来读取的操作会立刻返回，但是若还没有其他进程打开FIFO 文件来读取，则写入的操作会返回ENXIO 错误代码。
    2、没有使用O_NONBLOCK 旗标时，打开FIFO 来读取的操作会等到其他进程打开FIFO文件来写入才正常返回。同样地，打开FIFO文件来写入的操作会等到其他进程打开FIFO 文件来读取后才正常返回。
 
     打开管道失败，一般要注意打开的方式，一般不能以读写方式打开，要么只读打开，要么只写打开
      一定是有读的管道打开才能有写的管道
 
     */
    mkfifo(FIFO,0777);//其实就是一个文件而已 必须使用八进制的0777
    pid_t  pid;
    char buffer[BSIZE];
    //模拟情景让父进程暂停1秒 子进程打开了读的管道（一定要先打开读管道否则无法操作） 接下来子进程休息2s 父进程开始写入数据 子进程等待完成后 读取管道数据
    if((pid=fork())>0) {//父进程写 关闭读的管道
        sleep(1);
        int fd=open(FIFO,O_WRONLY|O_NONBLOCK);
        printf("文件描述符:%d\n",fd);
        memset(buffer,'\0',BSIZE);
        char* writeChar="hello my 儿子";
        memcpy(buffer,writeChar, strlen(writeChar));
        write(fd,buffer,BSIZE);
        close(fd);
        wait(NULL);
    }else{
        int fd=open(FIFO,O_RDONLY|O_NONBLOCK);
        sleep(2);
        printf("文件描述符:%d\n",fd);
        read(fd,buffer,BSIZE);
        printf("子进程读取到数据:%s\n",buffer);
        close(fd);
    }
}
void handler(int sig){
    if(sig==SIGINT){
        printf("接受到中断指令");
    }
}
/**
 * （3）、信号（signal）：信号是比较复杂的通信方式，用于通知接收进程有某种事件发生了，除了进程间通信外，进程还可以发送信号给进程本身；
 *   linux除了支持Unix早期信号语义函数sigal外，还支持语义符合Posix.1标准的信号函数sigaction（实际上，该函数是基于BSD的，BSD为了实现可靠信号机制，
 *  又能够统一对外接口，用sigaction函数重新实现了signal函数）。
 *  ctrl-c 是发送 SIGINT 信号，终止一个进程；进程无法再重续。
    ctrl-z 是发送 SIGSTOP信号，挂起一个进程；进程从前台转入后台并暂停，可以用bg使其后台继续运行，fg使其转入前台运行。
    ctrl-\ 发送 SIGQUIT 信号给前台进程组中的所有进程，终止前台进程并生成 core 文件。
 
    SIGINT
      程序终止(interrupt)信号, 在用户键入INTR字符(通常是Ctrl-C)时发出，用于通知前台进程组终止进程。
    SIGQUIT
    和SIGINT类似, 但由QUIT字符(通常是Ctrl-\)来控制. 进程在因收到SIGQUIT退出时会产生core文件, 在这个意义上类似于一个程序错误信号。
  1 SIGTERM
    程序结束(terminate)信号, 与SIGKILL不同的是该信号可以被阻塞和处理。通常用来要求程序自己正常退出，shell命令kill缺省产生这个信号。如果进程终止不了，我们才会尝试SIGKILL。
   SIGSTOP
    停止(stopped)进程的执行. 注意它和terminate以及interrupt的区别:该进程还未结束, 只是暂停执行. 本信号不能被阻塞, 处理或忽略.
 
   模拟父线程等待信号 子线程发送一个信号 或者在linux使用 kill -s SIGINT 进程号 或者 ctrl+c中断
 */
void signTest(){
    pid_t  pid;
    char buffer[BSIZE];
    if((pid=fork())>0) {//父进程写 关闭读的管道
        signal(SIGINT,handler);//SIGINT信号绑定到这个函数 自动回调
        pause();//暂停等待信号
    }else{
        sleep(1);
        kill(getppid(),SIGINT);
    }
 
}
 
struct MsgS{
    long mtype;
    char data[100];
};
/**
 * （4）、消息队列：消息队列是消息的链接表，包括Posix消息队列system V消息队列。有足够权限的进程可以向队列中添加消息，
 * 被赋予读权限的进程则可以读走队列中的消息。消息队列克服了信号承载信息量少，管道只能承载无格式字节流以及缓冲区大小受限等缺
 *
 */
void msgQueueTest(){
    //char* path="/bkey"
    //creat(path,0777);
    //key_t  k=ftok(path,2);//文件必须存在才能创建成功key
    //printf("获取到的key:%d\n",k);
    struct MsgS msg;
    //这里重点注意 msgget  cygwin没有实现 返回 88的errorno 去linux下编译运行才能正常 否则qid一直是-1
    int qid=msgget(IPC_PRIVATE,IPC_CREAT|0644);//msg.h头文件中定义  ftok产生一个或者使用IPC_PRIVATE
    printf("错误number :%d\n",errno);
    printf("消息队列编号:%d\n",qid);//必须是大于0的数字才是创建成功的
    pid_t  pid;
    char buffer[BSIZE];
    if((pid=fork())>0) {//父进程写
        sleep(2);
        //发送两个消息
        msg.mtype=2;//注意这里定义类型为2
        strcpy(msg.data,"hello child");
        int flag =msgsnd(qid,&msg,sizeof(struct MsgS),0);
        if ( flag < 0 )
        {
            perror("send message error") ;
            return  ;
        }
        struct MsgS msg1;
        msg1.mtype=1;//这里类型是1
        strcpy(msg1.data,"hello child1");
        msgsnd(qid,&msg1,sizeof(struct MsgS),0);
        if ( flag < 0 )
        {
            perror("send message error") ;
            return  ;
        }
    }else{
        //指定接受类型为1的所以 输出hello child1
        msgrcv(qid,&msg,sizeof(struct MsgS),1,0);//看其他人写的减去类型long的长度 sizeof(struct MsgS) -sizeof(long) 不明白 减不减都ok
        printf("接受到消息:%s\n",msg.data);
 
    }
}
 
/**
 * （5）、共享内存：使得多个进程可以访问同一块内存空间，是最快的可用IPC形式。是针对其他通信机制运行效率较低而设计的。
 * 往往与其它通信机制，如信号量结合使用，来达到进程间的同步及互斥。
 */
void testShareMemory(){
    char path[200];
#ifdef __linux
    strcpy(path,"/mykey");
#endif
#ifdef WINVER
    strcpy(path,"c:/mykey");
#endif
    printf("路径是:%s\n",path);
    creat(path,0644);
    key_t kt;
    kt=ftok(path,2);
    //创建一片共享内存 大小:100
    int shmid=shmget(kt,100,IPC_CREAT|0644);//在头文件 sys/shm.h  cygwin未实现
    printf("key是:%d，共享内存id:%d\n",kt,shmid);
    printf("错误编号:%d\n",errno);
    pid_t  pid;
    char buffer[BSIZE];
 
    if((pid=fork())>0) {
        //shmat函数的作用就是用来启动对该共享内存的访问，并把共享内存连接到当前进程的地址空间 返回共享内存的第一个位置的指针
        void* addr=shmat(shmid,NULL,0);
        strcpy((char*)addr,"hello child");
    }else{
        sleep(2);
        void* addr=shmat(shmid,NULL,0);
        printf("共享数据：%s\n",(char*)addr);
    }
}
```

 
》》线程

 
线程是可独立运行的单元 相对于进程来说 更加轻量级 优点如下：

 
在开销方面：每个进程都有独立的代码和数据空间（程序上下文），程序之间的切换会有较大的开销；线程可以看做轻量级的进程，同一类线程共享代码和数据空间，每个线程都有自己独立的运行栈和程序计数器（PC），线程之间切换的开销小。

 
内存分配方面：系统在运行的时候会为每个进程分配不同的内存空间；而对线程而言，除了CPU外，系统不会为线程分配内存（线程所使用的资源来自其所属进程的资源），线程组之间只能共享资源。

 
包含关系：没有线程的进程可以看做是单线程的，如果一个进程内有多个线程，则执行过程不是一条线的，而是多条线（线程）共同完成的；线程是进程的一部分，所以线程也被称为轻权进程或者轻量级进程。
  

 
```
#include  <stdio.h>
#include <stdlib.h>
#include "string.h"
#include <sys/unistd.h>
#include <pthread.h>
 
#define BSIZE 200
#define FIFO "c:/alink"
 
void* myfun(void * a){
    printf("线程被执行\n");
}
/**
 * linux编译时 指定 pthread的静态库 gcc th.c -lpthread -o th
 * clion中不需要链接静态库
 * @param p
 * @param arg
 * @return
 */
 int main(int p,char* arg[]) {
    pthread_t pt;
    int err=pthread_create(&pt,NULL,myfun,NULL);
    printf("线程id:%d\n",pt);
    if(err != 0){
         printf("can't create thread: %s\n",strerror(err));
         return 1;
    }
    //pthread_cancel(pt); 取消线程
    //pthread_join(pt,NULL);等待线程执行完成后 在往后执行
    sleep(2);
 }
```

 
网络编程 

 
定义一个socket服务 监听 8888端口使用tenlnet连接发送数据

 
```
#include  <stdio.h>
#include <stdlib.h>
#include "string.h"
#include "errno.h"
#include "errno.h"
#include <sys/unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "arpa/inet.h";
#define BSIZE 200
#define FIFO "c:/alink"
 
void* myfun(void * a){
    printf("线程被执行\n");
}
/**
 * linux编译时 指定 pthread的静态库 gcc th.c -lpthread -o th
 * clion中不需要链接静态库
 * @param p
 * @param arg
 * @return
 */
 int main(int p,char* arg[]) {
 
     //系统申请socket 返回sockfd文件描述符
     int sockfd=socket(AF_INET,SOCK_STREAM,0);//定义在#include <sys/socket.h>
     struct sockaddr_in servaddr;//定义在<netinet/in.h>
     servaddr.sin_addr.s_addr=htonl(INADDR_ANY);//绑定本机所有ip
     servaddr.sin_port=htons(8888);//绑定端口
     servaddr.sin_family=AF_INET;
     //将端口绑定到sock
     if(bind(sockfd,&servaddr, sizeof(servaddr))<0){
         perror(strerror(errno));
     }
     //开始监听 第二个参数允许最大连接数
     if(listen(sockfd,10)<0){
         perror(strerror(errno));
     }
    //接受请求
    while(1){
        struct sockaddr_in client;
        socklen_t len=sizeof(client);
        int connfd=accept(sockfd,( struct sockaddr_in* )&client,(socklen_t*)&len);//会阻塞 等待一个客户端连接
        printf("%s\n",strerror(errno));
        printf("接受到连接:%d 客户端的ip:%s",connfd,inet_ntoa(client.sin_addr));//inet_ntoa在 #include "arpa/inet.h";
        fflush(stdout);
        char buffer[ 10 ];
        ssize_t rsize=recv(connfd,buffer,9,0 );
        printf("接受到消息:%s 接受到的长度是：%d",buffer,rsize);
        fflush(stdout);
        sleep(1);
    }
    close(sockfd);
 
 }
```