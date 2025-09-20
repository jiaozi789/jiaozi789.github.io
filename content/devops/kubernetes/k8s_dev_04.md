---
title: "K8S二次开发04-自定义operator（operator-sdk调试）"
date: 2025-09-18T16:55:17+08:00
weight: 3
# bookComments: false
# bookSearchExclude: false
---


Operator 是 Kubernetes 的扩展软件，它利用 定制资源 管理应用及其组件。 Operator 遵循 Kubernetes 的理念，特别是在控制器 方面。
Operator操作那些有状态的基础设施服务，包括：组件升级、节点恢复、调整集群规模。一个理想化的运维平台必须是Operator自己维护有状态应用，并将人工干预降低到最低限度

# 什么是Operator？
为了理解什么是Operator，让我们先复习一下Kubernetes。Kubernetes实际是期望状态管理器。先在Kubernetes中指定应用程序期望状态（实例数，磁盘空间，镜像等），然后它会尝试把应用维持在这种状态。Kubernetes的控制平面运行在Master节点上，它包含数个controller以调和应用达到期望状态：
- 检查当前的实际状态（Pod、Deployment等）
- 将实际状态与spec期望状态进行比较
- 如果实际状态与期望状态不一致，controller将会尝试协调实际状态以达到一致

比如，通过RS定义Pod拥有3个副本。当其中一个Pod down掉时，Kubernetes controller通过wacth API发现期望运行3个副本，而实际只有2个副本在运行。于是它新创建出一个Pod实例。
## controller作用
如图所示，controller在Kubernetes中发挥的作用。
![在这里插入图片描述](/docs/images/content/devops/kubernetes/k8s_dev_04.md.images/fecdc107d98dda9ad50ecd4d79b7d9fb.png)
- 通过Kubectl命令发送对象spec定义（Pod，Deployment等）到Kubernetes Master节点的API服务
- Master节点调度对象运行
- 一旦对象运行，controller会持续检查对象并根据spec协调实际情况

通过这种方式，Kubernetes非常适合维护无状态应用。但它本身的资源类型（Pod，Deployments，Namespaces，Services，DaemonSets等）比较有限。虽然每种资源类型都预定了行为及协调方式，但它们的处理方式没有多大差别。
现在，如果您的应用更复杂并需要执行自定义操作以达到期望的运行状态，应该怎么办？

举一个有状态应用的例子。假如一个数据库应用运行在多个节点上。如果超过半数的节点出现故障，则需要按照特定步骤从特定快照中加载数据。使用原生Kubernetes对象类型和controller则难以实现。或者为有状态应用程序扩展节点，升级到新版本或灾难恢复。这些类型的操作通常需要非常具体的步骤，并且通常需要手动干预。

## controller系统结构
使用 CRD 定制资源后，仅仅是让 Kubernetes 能够识别定制资源的身份。创建定制资源实例后，Kubernetes 只会将创建的实例存储到数据库中，并不会触发任何业务逻辑。在  数据库保存定制资源实例是没有意义的，如果需要进行业务逻辑控制，就需要创建控制器。

Controller 的作用就是监听指定对象的新增、删除、修改等变化，并针对这些变化做出相应的响应，关于 Controller 的详细设计，可以参考 Harry (Lei) Zhang 老师在 twitter 上的分享，基本架构图如下：
![在这里插入图片描述](/docs/images/content/devops/kubernetes/k8s_dev_04.md.images/39fafb0ece4b6fe6e8933edf666179ed.png)
图中可看出，定制资源实例的变化会通过 Informer 存入 WorkQueue，之后 Controller 会消费 WorkQueue，并对其中的数据做出业务响应。

Operator 其实就是图中除了 API Server 和 etcd 的剩余部分。由于 Client、Informer 和 WorkQueue 是高度相似的，所以有很多项目可以自动化生成 Controller 之外的业务逻辑（如 Client、Informer、Lister），因此用户只需要专注于 Controller 中的业务逻辑即可。

## Operator自定义controller
Operator通过扩展Kubernetes定义Custom Controller，观察应用并根据实际状态执行自定义任务。应用被定义为Kubernetes对象：Custom Resource （CR），它包含yaml spec和被API服务接受对象类型（K8s kind）。这样，您可以在自定义规范中定义要观察的任何特定条件，并在实例与规范不匹配时协调实例。虽然Operator controller主要使用自定义组件，但它与原生Kubernetes controller协调方式非常类似。
![在这里插入图片描述](/docs/images/content/devops/kubernetes/k8s_dev_04.md.images/b9044ab1fdbf06a90a97ce06b8d82c4c.png)

Operator通过Custom Controller协调应用spec。虽然API服务知道Custom Controller，但Operator可以独立运行在集群内部或外部。

## 构建Operator
为了创建自定义Operator，我们需要如下资源：
1. Custom Resource（CR）spec，定义我们要观测的应用对象，以及为CR定义的API
2. Custom Controller，用来观测CR
3. Custom code，决定Custom Controller如何协调CR
4. Operator，管理Custom Controller
5. Deployment，定义Operator和自定义资源

所有上述内容都可以通过手工编写Go代码和spec，或通过kubebuilder等工具生成Kubernetes API。但最简单的方式（也是我们在这里使用的方法）是使用CoreOS operator-sdk为这些组件生成模版。它允许您通过CLI命令生成spec、controller以及Operator框架。一旦生成后，您可以在spec中定义自定义字段并编写协调的自定义代码。我们将在本教程的下一部分中展开介绍。

## 编写你自己的 Operator
如果生态系统中没可以实现你目标的 Operator，你可以自己编写代码。

你还可以使用任何支持 Kubernetes API 客户端 的语言或运行时来实现 Operator（即控制器）。

以下是一些库和工具，你可用于编写自己的云原生 Operator。

说明： 本部分链接到提供 Kubernetes 所需功能的第三方项目。Kubernetes 项目作者不负责这些项目。此页面遵循CNCF 网站指南，按字母顺序列出项目。要将项目添加到此列表中，请在提交更改之前阅读内容指南。
[Charmed Operator Framework](https://juju.is/)
[kubebuilder](https://book.kubebuilder.io/)
[KubeOps (dotnet operator SDK)](https://buehler.github.io/dotnet-operator-sdk/)
[KUDO (Kubernetes 通用声明式 Operator)](https://kudo.dev/)
[Metacontroller，可与 Webhooks 结合使用，以实现自己的功能。](https://metacontroller.github.io/metacontroller/intro.html)
[Operator Framework](https://operatorframework.io/)
[shell-operator](https://github.com/flant/shell-operator)

# 使用operator sdk编写
operator sdk项目是Operator Framework的一个组件，这是一个开源工具包，以有效，自动化和可扩展的方式管理Kubernetes原生应用程序，称为Operators。更多介绍内容，请阅读博客。
Operators 可以在Kubernetes之上轻松地管理复杂有状态的应用程序。然而，由于诸如使用低级API，编写样板以及缺乏模块导致重复性工作等挑战，导致目前编写Operator可能很困难。
Operator SDK是一个框架，旨在简化Operator的编写，它提供如下功能：
- 高级API和抽象，更直观地编写操作逻辑
- 用于脚手架和代码生成的工具，可以快速引导新项目
- 扩展以涵盖常见的操作员用例

## 工作流程
SDK提供以下工作流程来开发新的Operator：
- 使用SDK命令行界面（CLI）创建新的Operator项目
- 通过添加自定义资源定义（CRD）定义新资源API
- 使用SDK API监控指定的资源
- 在指定的处理程序中定义Operator协调逻辑(对比期望状态与实际状态)，并使用SDK API与资源进行交互
- 使用SDK CLI构建并生成Operator部署manifests
Operator使用SDK在用户自定义的处理程序中以高级API处理监视资源的事件，并采取措施来reconcile（对比期望状态与实际状态）应用程序的状态。

## 快速开始
先给出我的环境，注意operator-sdk支持通过代码安装，在window上通过idea等工具搭配golang开发，有环境的可直接使用macos。

```cpp
开发操作系统：window
k8s: 1.23.3
k8s部署：debian|dockerdesktop自带的k8s
```
### 前置安装
-  安装gcc和make。
-  安装golang1.17以上版本。
-  一个可进入的公共的docker registry服务，并且准备一个域名作为registry服务的域名。

#### 安装gcc和make
```cpp
apt-get install gcc automake autoconf libtool make
```
window下直接安装cygwin，选择gcc和make等组件即可，我已经提前安装，如果不知道怎么安装可参考：https://blog.csdn.net/liaomin416100569/article/details/105127557?spm=1001.2014.3001.5501。
#### golang
安装 golang 17以上版本

```cpp
wget https://studygolang.com/dl/golang/go1.17.linux-amd64.tar.gz
tar zxvf go1.17.linux-amd64.tar.gz 
```
我这直接解压在/root下，/etc/profile添加环境变量

```cpp
export GOPATH=/root/go
export GOROOT=${GOPATH}
export GOARCH=386
export GOOS=linux
export GOTOOLS=$GOROOT/pkg/tool
export PATH=$PATH:$GOROOT/bin:$GOPATH/bin
```
让配置生效 并修改golang的私服
```cpp
source /etc/profile && go env -w GOPROXY=https://mirrors.aliyun.com/goproxy/
```
检测是否安装成功
```cpp
root@liaok8s:~/go# go version                   
go version go1.17 linux/amd64
root@liaok8s:~/go# go env | grep GOPROXY
GOPROXY="https://mirrors.aliyun.com/goproxy/"
```
#### 安装docker
开发本机需安装docker，operator的controller需要生成docker镜像
linux机器使用 yum或者apt-get安装 docker-ce即可
window安装docker desktop即可。

#### 安装docker registry
安装可用的docker registry | nexus | harbor
harbor安装参考：https://blog.csdn.net/liaomin416100569/article/details/86599571
这里直接选择docker registry (我的ip：10.10.0.115，后续设置一个本地域名:jiaozi.com)
>这里为了简单，就不设置账号密码权限了。
```cpp
docker run -d -p 5000:5000 --name registry --restart=always -v /opt/registry/data:/var/lib/registry docker.io/registry
```
检测存在的镜像

```cpp
 curl http://127.0.0.1:5000/v2/_catalog
```
在开发的window和k8s的worker节点 linux(/etc/docker/daemon.json)设置，改后重启;

```cpp
  "insecure-registries": [
    "jiaozi.com:5000"
  ],
```
在开发的本地机器和k8s的worker节点绑定hosts

```cpp
正在 Ping jiaozi.com [10.10.0.115] 具有 32 字节的数据:
来自 10.10.0.115 的回复: 字节=32 时间<1ms TTL=63
来自 10.10.0.115 的回复: 字节=32 时间<1ms TTL=63
来自 10.10.0.115 的回复: 字节=32 时间<1ms TTL=63

10.10.0.115 的 Ping 统计信息:
    数据包: 已发送 = 3，已接收 = 3，丢失 = 0 (0% 丢失)，
往返行程的估计时间(以毫秒为单位):
    最短 = 0ms，最长 = 0ms，平均 = 0ms
```
测试上传一个镜像

```cpp
docker pull nginx && docker tag nginx jiaozi.com:5000/nginx && docker push jiaozi.com:5000/nginx  && curl jiaozi.com/5000/v2/_catalog
{"repositories":["nginx"]}
```
#### 安装kubectl
让开发的机器可以通过kubectl访问集群，一般用window开发的 话，新版的docker desktop都带上内置的k8s，自然自带kubectl命令，可将远程k8s集群的初始化时生成的config文件拷贝到出来

```cpp
Your Kubernetes control-plane has initialized successfully!

To start using your cluster, you need to run the following as a regular user:

  mkdir -p $HOME/.kube
  sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
  sudo chown $(id -u):$(id -g) $HOME/.kube/config

Alternatively, if you are the root user, you can run:

  export KUBECONFIG=/etc/kubernetes/admin.conf
```
一般就是/etc/kubernetes/admin.conf，放到window的  %USERPROFILE%/.kube/config文件即可

```cpp
C:\Users\liaomin>echo %USERPROFILE%
C:\Users\liaomin mkdir %USERPROFILE%\.kube
将admin.conf重命名为config文件即可
```

![在这里插入图片描述](/docs/images/content/devops/kubernetes/k8s_dev_04.md.images/632aea0b9b437bfa7bb96459825c8f80.png)
测试

```cpp
C:\Users\liaomin\.kube>kubectl get pods
NAME       READY   STATUS    RESTARTS        AGE
dnsutils   1/1     Running   123 (22m ago)   5d2h
nginx      1/1     Running   1 (22m ago)     5d3h
```

### 安装operator-sdk
安装过程请参考：https://sdk.operatorframework.io/docs/installation/
非macos建议源代码安装，window的话先安装cygwin，打开cygwin Terminal 执行命令。
> window下升级go  go get golang.org/dl/go<version> && go<version> download  下载完成后，window下直接安装在，%USERPROFILE%\sdk 目录下，可直接修改path到 %USERPROFILE%\sdk\go<version>\bin目录下。
> 
```cpp
git clone https://github.com/operator-framework/operator-sdk
cd operator-sdk
git checkout master
make install
```
安装完成后 
```cpp
root@liaok8s:~/go# operator-sdk version
operator-sdk version: "v1.17.0-3-g158da144-dirty", commit: "158da1444dc400bcc7402e4e47787d56a9b4f4ad", kubernetes version: "v1.21", go version: "go1.17", GOOS: "linux", GOARCH: "amd64"
```
>window安装完成后目录为：
$ which operator-sdk
/cygdrive/c/Users/liaomin/go/bin/operator-sdk

### operator-sdk实例
以下实例参考自官方文档：https://sdk.operatorframework.io/docs/building-operators/golang/tutorial/
功能实现如下：
将创建一个简单的案例项目
1. 如果不存在即创建一个nginx Deployment 
2. 确保deploy的大小和CR文件制定的大小一致
3. 更新nginx CR的状态（同步名字相同的PODS状态） 
#### 创建项目
初始化一个新的项目包含以下内容 :

-  go.mod  项目依赖。
-  PROJECT 存储项目的配置。
-  Makefile  一些常用的make targets。
- config目录下许多YAML files 用于项目发布。
- main.go 用于创建 manager，运行项目controllers。
>更加详细的目录结果信息可查看kubebuilder文档：https://book.kubebuilder.io/cronjob-tutorial/basic-project.html
```cpp
liaomin@DESKTOP-FSEDE3P /cygdrive/d/code1/helloworld-operator
$ operator-sdk init --domain jiaozi.com --repo github.com/lzeqian/helloworld-operator
Writing kustomize manifests for you to edit...
Writing scaffold for you to edit...
Get controller runtime:
$ go get sigs.k8s.io/controller-runtime@v0.11.0
Update dependencies:
$ go mod tidy
Next: define a resource with:
$ operator-sdk create api
```
#### 创建api
通过脚手架生成一个crd和controller的api，没有没有指定--resource --controller，将会通过交互通过用户确认是否生成，命令执行时相关的golang依赖将自动下载安装。
```cpp
operator-sdk create api --group test --version v1alpha1 --kind HelloWorld --resource --controller
```
生成了一个api和controller目录包含api的结构体定义和controller代码。
如果使用cygwin terminal去执行一般都会报错

```cpp
$ operator-sdk create api --group test --version v1alpha1 --kind HelloWorld --resource --controller
Writing kustomize manifests for you to edit...
Writing scaffold for you to edit...
api\v1alpha1\helloworld_types.go
controllers\helloworld_controller.go
Update dependencies:
$ go mod tidy
Running make:
$ make generate
go: creating new go.mod: module tmp
Downloading sigs.k8s.io/controller-tools/cmd/controller-gen@v0.8.0
go get: installing executables with 'go get' in module mode is deprecated.
        To adjust and download dependencies of the current module, use 'go get -d'.
        To install using requirements of the current module, use 'go install'.
        To install ignoring the current module, use 'go install' with a version,
        like 'go install example.com/cmd@latest'.
        For more information, see https://golang.org/doc/go-get-install-deprecation
        or run 'go help get' or 'go help install'.
cannot install, GOBIN must be an absolute path
make: *** [Makefile:142：controller-gen] 错误 1
Error: failed to create API: unable to run post-scaffold tasks of "base.go.kubebuilder.io/v3":
 exit status 2

```
通过错误可以知道是在执行make generate时下载sigs.k8s.io/controller-tools/cmd/controller-gen@v0.8.0报错，错误是GOBIN must be an absolute path，说明GOBIN对应的路径是错的导致下载的包无法写入，通过查看Makefile的generate任务会调用一个自定义函数go-get-tool

```cpp
# go-get-tool will 'go get' any package $2 and install it to $1.
PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
define go-get-tool
@[ -f $(1) ] || { \
set -e ;\
TMP_DIR=$$(mktemp -d) ;\
cd $$TMP_DIR ;\
go mod init tmp ;\
echo "Downloading $(2) $(PROJECT_DIR)" ;\
GOBIN=$(PROJECT_DIR)/bin go get $(2) ;\
rm -rf $$TMP_DIR ;\
}
endef
```
发现其中一句：GOBIN=$(PROJECT_DIR)/bin go get $(2) ;\
设置GOBIN=当前项目目录/bin目录然后执行 go get命令，打印GOBIN发现路径是：/cygdrive/d/code1/helloworld-operator，对于goget来说程序内部肯定是不能识别这个路径，但是命令行是可以识别的，比如mkdir  /cygdrive/d/code1/helloworld-operator/aa是可行的，我只需要将
PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST)))) 替换为实际的目录即可
修改PROJECT_DIR=你项目的根目录即可
```cpp
# go-get-tool will 'go get' any package $2 and install it to $1.
PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIR := D:/code1/helloworld-operator

define go-get-tool
@[ -f $(1) ] || { \
set -e ;\
TMP_DIR=$$(mktemp -d) ;\
cd $$TMP_DIR ;\
go mod init tmp ;\
echo "Downloading $(2) $(PROJECT_DIR)" ;\
GOBIN=$(PROJECT_DIR)/bin go get $(2) ;\
rm -rf $$TMP_DIR ;\
}
endef
```
修改完成后正常执行，并且根目录生成了bin目录，并且下载了controller-gen
![在这里插入图片描述](/docs/images/content/devops/kubernetes/k8s_dev_04.md.images/8217f75b3deb3628b97d0326c5aac294.png)
修改api/v1alpha1/helloworld_types.go 文件

```cpp
package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

/**
最后发布后的cr就是这样
kubectl get memcached/memcached-sample -o yaml
apiVersion: cache.example.com/v1alpha1
kind: Memcached
metadata:
  clusterName: ""
  creationTimestamp: 2018-03-31T22:51:08Z
  generation: 0
  name: memcached-sample
  namespace: default
  resourceVersion: "245453"
  selfLink: /apis/cache.example.com/v1alpha1/namespaces/default/memcacheds/memcached-sample
  uid: 0026cc97-3536-11e8-bd83-0800274106a1
spec:
  size: 3
status:
  nodes:
  - memcached-sample-6fd7c98d8-7dqdr
  - memcached-sample-6fd7c98d8-g5k7v
  - memcached-sample-6fd7c98d8-m7vn7
 */

// HelloWorldSpec 定义上面spec的部分
type HelloWorldSpec struct {
	//对应crd自定义字段 spec.size
	Size int32 `json:"size"`
}

// HelloWorldStatus定义HelloWorld的状态观察
type HelloWorldStatus struct {
	Nodes []string `json:"nodes"`
}
//表示下面这个结构是yaml的根 同时添加子资源状态和scale
//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
// HelloWorld是 helloworlds API的定义
type HelloWorld struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   HelloWorldSpec   `json:"spec,omitempty"`
	Status HelloWorldStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// HelloWorldList 包含多个 HelloWorld
type HelloWorldList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []HelloWorld `json:"items"`
}

func init() {
	SchemeBuilder.Register(&HelloWorld{}, &HelloWorldList{})
}
```
执行命令

```cpp
make generate
```
>该命令会执行controller-gen 工具更新 api/v1alpha1/zz_generated.deepcopy.go

执行命令

```cpp
make manifests
```
>将执行 controller-gen生成 CRD文件： config/crd/bases/test.jiaozi.com_helloworlds.yaml

生成的crd文件和定义的go代码是一致的

```cpp
---
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  annotations:
    controller-gen.kubebuilder.io/version: v0.8.0
  creationTimestamp: null
  name: helloworlds.test.jiaozi.com
spec:
  group: test.jiaozi.com
  names:
    kind: HelloWorld
    listKind: HelloWorldList
    plural: helloworlds
    singular: helloworld
  scope: Namespaced
  versions:
  - name: v1alpha1
    schema:
      openAPIV3Schema:
        description: 表示下面这个结构是yaml的根 同时添加子资源状态和scale HelloWorld是 helloworlds API的定义
        properties:
          apiVersion:
            description: 'APIVersion defines the versioned schema of this representation
              of an object. Servers should convert recognized schemas to the latest
              internal value, and may reject unrecognized values. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#resources'
            type: string
          kind:
            description: 'Kind is a string value representing the REST resource this
              object represents. Servers may infer this from the endpoint the client
              submits requests to. Cannot be updated. In CamelCase. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds'
            type: string
          metadata:
            type: object
          spec:
            description: HelloWorldSpec 定义上面spec的部分
            properties:
              size:
                description: 对应crd自定义字段 spec.size
                format: int32
                type: integer
            required:
            - size
            type: object
          status:
            description: HelloWorldStatus定义HelloWorld的状态观察
            properties:
              nodes:
                items:
                  type: string
                type: array
            required:
            - nodes
            type: object
        type: object
    served: true
    storage: true
    subresources:
      status: {}
status:
  acceptedNames:
    kind: ""
    plural: ""
  conditions: []
  storedVersions: []

```
#### 实现controller

开发过程中使用的api接口包参考：
1. corev1 "k8s.io/api/core/v1" 核心api，提供核心结构和接口，yaml中常用的Spec定义在此。
2. metav1 "k8s.io/apimachinery/pkg/apis/meta/v1" yaml中常用的metadata定义，ObjectMeta,LabelSelector等基本在此。
3. appsv1 "k8s.io/api/apps/v1" 常用的创建的crd或者已经存在的rd等都在此，比如Deployments,Pod,Service等等。

实现controller代码

```cpp
package controllers

import (
	"context"
	"github.com/go-logr/logr"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"reflect"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	testv1alpha1 "github.com/lzeqian/helloworld-operator/api/v1alpha1"
)

// HelloWorldReconciler reconciles a HelloWorld object
type HelloWorldReconciler struct {
	client.Client
	Log    logr.Logger //日志打印
	Scheme *runtime.Scheme
}

//+kubebuilder:rbac:groups=test.jiaozi.com,resources=helloworlds,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=test.jiaozi.com,resources=helloworlds/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=test.jiaozi.com,resources=helloworlds/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the HelloWorld object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.11.0/pkg/reconcile
func (r *HelloWorldReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	_ = log.FromContext(ctx)
	r.Log.Info("recondile被调用" + req.Namespace + "-" + req.Name)
	// TODO(user): your logic here
	helloworld := &testv1alpha1.HelloWorld{}
	err := r.Client.Get(ctx, req.NamespacedName, helloworld)
	if err != nil {
		//如果是找不到异常 说明这个cr已经被删除了
		if errors.IsNotFound(err) {
			r.Log.Info("crd资源已经被删除")
			//停止loop循环不在订阅事件。
			return ctrl.Result{}, nil
		}
		//返回错误，但是继续监听事件
		return ctrl.Result{}, err
	}
	//找到了cr就可以确认cr下的deploy是否存在
	nginxDeployFound := &appsv1.Deployment{}
	//获取当前namespace下的deploy
	errDeploy := r.Client.Get(ctx, types.NamespacedName{Name: helloworld.Name, Namespace: helloworld.Namespace}, nginxDeployFound)
	if errDeploy != nil {
		//不存在，需要创建
		if errors.IsNotFound(errDeploy) {
			r.Log.Info("不存在deploy，新建deploy")
			//类似于yaml的语法创建ngxindeploy
			nginxDeploy := &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      helloworld.Name,
					Namespace: helloworld.Namespace,
				},
				Spec: appsv1.DeploymentSpec{
					Replicas: &helloworld.Spec.Size,
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"hello_name": helloworld.Name,
						},
					},
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"hello_name": helloworld.Name,
							},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Image: "nginx",
									Name:  "nginx",
									Ports: []corev1.ContainerPort{{
										ContainerPort: 80,
										Name:          "nginx",
									}},
								},
							},
						},
					},
				},
			}
			controllerutil.SetControllerReference(helloworld, nginxDeploy, r.Scheme)
			if err1 := r.Client.Create(ctx, nginxDeploy); err1 != nil {
				r.Log.Info("不存在deploy，新建deploy失败")
				return ctrl.Result{}, errDeploy
			}
			r.Log.Info("不存在deploy，新建deploy成功")
			return ctrl.Result{Requeue: true}, nil
		} else {
			return ctrl.Result{}, errDeploy
		}
	}
	//如果err是空的说明找到了一个已经存在的deploy,需要判断deploy实际的个数和预期crd上的个数是否一致的
	if *nginxDeployFound.Spec.Replicas != helloworld.Spec.Size {
		r.Log.Info("deploy对应pod数量错误，更新deploy为helloword的size")
		//修改原始的对象的spec.replicas
		nginxDeployFound.Spec.Replicas = &helloworld.Spec.Size
		//更新deploy
		if err = r.Update(ctx, nginxDeployFound); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}
	//更新找到的deploy的pod的数量更新到helloworld的status.nodes上
	podList := &corev1.PodList{}
	listOpts := []client.ListOption{
		client.InNamespace(helloworld.Namespace),
		client.MatchingLabels(map[string]string{
			"hello_name": helloworld.Name,
		}),
	}
	if err = r.List(ctx, podList, listOpts...); err != nil {
		return ctrl.Result{}, err
	}

	// 更新pod的实际个数的名字写入到helloworld的status上
	podNames := []string{}
	for pn := range podList.Items {
		podNames = append(podNames, podList.Items[pn].Name)
	}
	if !reflect.DeepEqual(podNames, helloworld.Status.Nodes) {
		helloworld.Status.Nodes = podNames
		r.Log.Info("更新状态多helloword的子status")
		if err := r.Status().Update(ctx, helloworld); err != nil {
			return ctrl.Result{}, err
		}
	}
	return ctrl.Result{}, nil
}

// for表示监控的cr的类型，Owns表示监控的第二资源也就是cr需要控制的资源
func (r *HelloWorldReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		//watch HelloWorld这个crd作为一监控资源
		For(&testv1alpha1.HelloWorld{}).
		//watch Deployment作为第二个监控资源
		Owns(&appsv1.Deployment{}).
		Complete(r)
}

```
#### 运行operator
运行operator支持以下三种方式。
1. controller作为一个go应用运行在集群之外方式运行。
2. 作为一个deployment直接将程序打包成镜像运行在集群内部。
3. 使用oolm方式部署。

>注意这里运行operator实际上是有3步
>1. 将crd的定义安装到集群。
>2. 运行controller监控cr的资源创建/删除/修改等，实际运行的程序就是运行controller，controller可以运行在集群外和集群内均可。
>3. 创建一个cr，触发controller。

其中2和3方式适合生产部署：可参考：https://master.sdk.operatorframework.io/docs/building-operators/golang/tutorial/#run-the-operator
这里演示第一种方式，因为方便调试，我这使用idea在window下环境。
前置条件：window安装kubectl参考前面的章节。
确保能正常执行kubectl命令

```cpp
D:\code1\helloworld-operator>kubectl get pods
NAME       READY   STATUS    RESTARTS        AGE
dnsutils   1/1     Running   290 (59m ago)   12d
nginx      1/1     Running   1 (7d ago)      12d
```
##### 修改配置
注意该配置修改，如果是本地调试可以不执行，如果使用2-3方式运行是必选项
配置镜像，修改Makefile
注释掉
```cpp
-IMG ?= controller:latest
```
修改为你自己的registry私服
```cpp
IMAGE_TAG_BASE ?= jiaozi.com:5000/helloworld-operator

# BUNDLE_IMG defines the image:tag used for the bundle.
# You can use it as an arg. (E.g make bundle-build BUNDLE_IMG=<some-registry>/<project-name-bundle>:<tag>)
BUNDLE_IMG ?= $(IMAGE_TAG_BASE)-bundle:v$(VERSION)
```
修改Dockerfile
注释掉代码
```cpp
#USER 65532:65532
```
将基础镜像
```cpp
FROM gcr.io/distroless/static:nonroot
```
替换为：
```cpp
FROM alpine:latest
```
去除单元测试
修改makefile
```cpp
docker-build: test ## Build docker image with the manager.
```
为
```cpp
docker-build:  ## Build docker image with the manager.
```

执行命令，构建镜像，推送到私服
```cpp
make docker-build docker-push
```
##### 打印日志
打印日志主要是方便调试。
main.go中创建Reconciler部分添加参数
```cpp
if err = (&controllers.HelloWorldReconciler{
		Client: mgr.GetClient(),
		Log:    ctrl.Log.WithName("helloworld"),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "HelloWorld")
		os.Exit(1)
	}
```
HelloWorldReconciler结构体定义添加Log
```cpp
type HelloWorldReconciler struct {
	client.Client
	Log    logr.Logger //日志打印
	Scheme *runtime.Scheme
}
```
可在方法中使用
- r.Log.Info("crd资源已经被删除")  打印info日志
- r.Log.Error("异常信息") 打印error日志

##### 安装crd
```cpp
make install
```
查看安装

```cpp
D:\code1\helloworld-operator>kubectl get crd | grep hello
helloworlds.test.jiaozi.com                           2022-02-28T10:30:42Z
```
查看定义
```cpp
D:\code1\helloworld-operator>kubectl explain helloworlds.test.jiaozi.com
KIND:     HelloWorld
VERSION:  test.jiaozi.com/v1alpha1

DESCRIPTION:
     表示下面这个结构是yaml的根 同时添加子资源状态和scale
     HelloWorld是 helloworlds API的定义

FIELDS:
   apiVersion   <string>
     APIVersion defines the versioned schema of this representation of an
     object. Servers should convert recognized schemas to the latest internal
     value, and may reject unrecognized values. More info:
     https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#resources

   kind <string>
     Kind is a string value representing the REST resource this object
     represents. Servers may infer this from the endpoint the client submits
     requests to. Cannot be updated. In CamelCase. More info:
     https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds

   metadata     <Object>
     Standard object's metadata. More info:
     https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata

   spec <Object>
     HelloWorldSpec 定义上面spec的部分

   status       <Object>
     HelloWorldStatus定义HelloWorld的状态观察

```
##### 运行controller
运行controller watch cr的创建/删除/修改
```cpp
D:\code1\helloworld-operator>make run .
/cygdrive/d/code1/helloworld-operator/bin/controller-gen rbac:roleName=manager-role crd webhook paths="./..." output:crd:artifacts:config=config/crd/bases
/cygdrive/d/code1/helloworld-operator/bin/controller-gen object:headerFile="hack\\boilerplate.go.txt" paths="./..."
go fmt ./...
go vet ./...
go run ./main.go
1.6460445940803254e+09  INFO    controller-runtime.metrics      Metrics server is starting to listen    {"addr": ":8080"}
1.6460445940819194e+09  INFO    setup   starting manager
1.6460445940819194e+09  INFO    Starting server {"kind": "health probe", "addr": "[::]:8081"}
1.6460445940819194e+09  INFO    Starting server {"path": "/metrics", "kind": "metrics", "addr": "[::]:8080"}
1.646044594082426e+09   INFO    controller.helloworld   Starting EventSource    {"reconciler group": "test.jiaozi.com", "reconciler kind": "HelloWorld", "source": "kind source: *v1alpha1.HelloWorld"}
1.6460445940830188e+09  INFO    controller.helloworld   Starting EventSource    {"reconciler group": "test.jiaozi.com", "reconciler kind": "HelloWorld", "source": "kind source: *v1.Deployment"}
1.6460445940830188e+09  INFO    controller.helloworld   Starting Controller     {"reconciler group": "test.jiaozi.com", "reconciler kind": "HelloWorld"}
1.6460445941841855e+09  INFO    controller.helloworld   Starting workers        {"reconciler group": "test.jiaozi.com", "reconciler kind": "HelloWorld", "worker count": 1}

```
在项目根目录创建一个cr 使用命令执行
```cpp
apiVersion: test.jiaozi.com/v1alpha1
kind: HelloWorld
metadata:
  name: helloworld-sample
spec:
  size: 2

```
kubectl apply -f helloworld.yaml
可看到controller的控制台出现了直接log打印的日志
```cpp
1.6460447152888484e+09  INFO    helloworld      recondile被调用default-helloworld-sample
1.6460447152893913e+09  INFO    helloworld      不存在deploy，新建deploy
1.6460447152947052e+09  INFO    helloworld      不存在deploy，新建deploy成功
1.646044715295705e+09   INFO    helloworld      recondile被调用default-helloworld-sample
1.6460447153962688e+09  INFO    helloworld      更新状态多helloword的子status
1.6460447154029741e+09  INFO    helloworld      recondile被调用default-helloworld-sample

```
查看deploy和pod的个数
```cpp
D:\code1\helloworld-operator>kubectl get deploy
NAME                READY   UP-TO-DATE   AVAILABLE   AGE
helloworld-sample   2/2     2            2           60s
D:\code1\helloworld-operator>kubectl get pod | grep hello
helloworld-sample-569448757-pdq7q   1/1     Running   0               75s
helloworld-sample-569448757-x4b9p   1/1     Running   0               75s

```
##### 调试controller
首先生成exe文件

```cpp
go build
```
![在这里插入图片描述](/docs/images/content/devops/kubernetes/k8s_dev_04.md.images/96cbf3398504182eefbb2edb8c24096a.png)
安装dlv，如果下载有错，多下载几次，最好设置代理
```cpp
go get -u github.com/go-delve/delve/cmd/dlv
```
~~打开idea help-edit custom properties，新增dlv参数位置（该步骤省略）：~~
```cpp
dlv.path=C:/Users/liaomin/go/bin/dlv.exe
```
执行dlv命令开放2345端口
```cpp
dlv exec --headless --listen ":2345" --log --api-version 2 ./helloworld-operator.exe
```
idea创建一个go remote
![在这里插入图片描述](/docs/images/content/devops/kubernetes/k8s_dev_04.md.images/d65859fe9c2b136964c60950be8c98bf.png)
host和port默认即可，确认后，程序开始正常运行，可以在controller下断点，执行kubectl新增/删除cr资源可正常调试
![在这里插入图片描述](/docs/images/content/devops/kubernetes/k8s_dev_04.md.images/7b252a370210fdd7b0fe25a875974695.png)
