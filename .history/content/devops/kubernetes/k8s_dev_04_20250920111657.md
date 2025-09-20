---
title: "K8S二次开发03-CRD资源详解"
date: 2025-09-18T16:55:17+08:00
weight: 3
# bookComments: false
# bookSearchExclude: false
---

# CustomResourceDefinition简介：
在 Kubernetes 中一切都可视为资源，Kubernetes 1.7 之后增加了对 CRD 自定义资源二次开发能力来扩展 Kubernetes API，通过 CRD 我们可以向 Kubernetes API 中增加新资源类型，而不需要修改 Kubernetes 源码来创建自定义的 API server，该功能大大提高了 Kubernetes 的扩展能力。
当你创建一个新的CustomResourceDefinition (CRD)时，Kubernetes API服务器将为你指定的每个版本创建一个新的RESTful资源路径，我们可以根据该api路径来创建一些我们自己定义的类型资源。CRD可以是命名空间的，也可以是集群范围的，由CRD的作用域(scpoe)字段中所指定的，与现有的内置对象一样，删除名称空间将删除该名称空间中的所有自定义对象。customresourcedefinition本身没有名称空间，所有名称空间都可以使用。
## 创建crd定义
[Kuberneters 官方文档](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/)  [中文版本](https://kubernetes.io/zh/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/)
通过crd资源创建自定义资源，即自定义一个Restful API：
```cpp
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  # 名称必须与下面的spec字段匹配，格式为: <plural>.<group>
  name: crontabs.stable.example.com
spec:
  # 用于REST API的组名称: /apis/<group>/<version>
  group: stable.example.com
  # 此CustomResourceDefinition支持的版本列表
  versions:
    - name: v1
      # 每个版本都可以通过服务标志启用/禁用。
      served: true
      # 必须将一个且只有一个版本标记为存储版本。
      storage: true
      #使用v3定义创建容器的属性 cronSpec和image是字符串类型replicas是int型
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                cronSpec:
                  type: string
                  description: "定时任务触发时间"
                image:
                  type: string
                  description: "镜像"
                replicas:
                  type: integer
                  description: "副本数"
  # 指定crd资源作用范围在命名空间或集群
  scope: Namespaced
  names:
    # URL中使用的复数名称: /apis/<group>/<version>/<plural>
    plural: crontabs
    # 在CLI(shell界面输入的参数)上用作别名并用于显示的单数名称
    singular: crontab
    # kind字段使用驼峰命名规则. 资源清单使用如此
    kind: CronTab
    # 短名称允许短字符串匹配CLI上的资源，意识就是能通过kubectl 在查看资源的时候使用该资源的简名称来获取。
    shortNames:
    - ct
```
>注意：这是只是一个资源定义，类似于java中class的定义，new class是创建的对象。
>在k8s中 pod，service都是已经预先定义的资源，通过kubectl create|run创建的是实例，定义只有一个，实例可以有多个。

具体的详细语法可以使用kube explain查看
```cpp
kubectl explain crd --recursive
```
在k8s中创建资源类型
```cpp
root@liaok8s:/home/mainte/coredns# kubectl apply -f crd.yaml 
customresourcedefinition.apiextensions.k8s.io/crontabs.stable.example.com created
root@liaok8s:/home/mainte/coredns# kubectl get crd
NAME                                                  CREATED AT
bgpconfigurations.crd.projectcalico.org               2022-02-14T09:06:15Z
bgppeers.crd.projectcalico.org                        2022-02-14T09:06:15Z
blockaffinities.crd.projectcalico.org                 2022-02-14T09:06:15Z
caliconodestatuses.crd.projectcalico.org              2022-02-14T09:06:15Z
clusterinformations.crd.projectcalico.org             2022-02-14T09:06:15Z
crontabs.stable.example.com                           2022-02-17T02:41:58Z
felixconfigurations.crd.projectcalico.org             2022-02-14T09:06:15Z
globalnetworkpolicies.crd.projectcalico.org           2022-02-14T09:06:15Z
globalnetworksets.crd.projectcalico.org               2022-02-14T09:06:15Z
hostendpoints.crd.projectcalico.org                   2022-02-14T09:06:15Z
ipamblocks.crd.projectcalico.org                      2022-02-14T09:06:15Z
ipamconfigs.crd.projectcalico.org                     2022-02-14T09:06:15Z
ipamhandles.crd.projectcalico.org                     2022-02-14T09:06:15Z
ippools.crd.projectcalico.org                         2022-02-14T09:06:15Z
ipreservations.crd.projectcalico.org                  2022-02-14T09:06:15Z
kubecontrollersconfigurations.crd.projectcalico.org   2022-02-14T09:06:15Z
networkpolicies.crd.projectcalico.org                 2022-02-14T09:06:15Z
networksets.crd.projectcalico.org                     2022-02-14T09:06:15Z
```
会自动创建一个新的带有名称空间的RESTful API端点:
/apis/stable.example.com/v1/namespaces/*/crontabs/...然后我们可以使用该url来创建和管理自定义对象资源。
查看api版本

```cpp
root@liaok8s:/home/mainte/coredns# kubectl api-versions | grep stable.example.com
stable.example.com/v1
```
查看api资源

```cpp
root@liaok8s:/home/mainte/coredns# kubectl api-resources | grep -E example
NAME                              SHORTNAMES   APIVERSION                             NAMESPACED   KIND
crontabs                          ct           stable.example.com/v1                  true         CronTab
```
查看整个api的yaml定义

```cpp
root@liaok8s:/home/mainte/coredns# kubectl explain ct --recursive    
KIND:     CronTab
VERSION:  stable.example.com/v1

DESCRIPTION:
     <empty>

FIELDS:
   apiVersion   <string>
   kind <string>
   metadata     <Object>
      annotations       <map[string]string>
      clusterName       <string>
      creationTimestamp <string>
      deletionGracePeriodSeconds        <integer>
      deletionTimestamp <string>
      finalizers        <[]string>
      generateName      <string>
      generation        <integer>
      labels    <map[string]string>
      managedFields     <[]Object>
         apiVersion     <string>
         fieldsType     <string>
         fieldsV1       <map[string]>
         manager        <string>
         operation      <string>
         subresource    <string>
         time   <string>
      name      <string>
      namespace <string>
      ownerReferences   <[]Object>
         apiVersion     <string>
         blockOwnerDeletion     <boolean>
         controller     <boolean>
         kind   <string>
         name   <string>
         uid    <string>
      resourceVersion   <string>
      selfLink  <string>
      uid       <string>
   spec <Object>
      cronSpec  <string>
      image     <string>
      replicas  <integer>
```
查看某个字段的定义

```cpp
root@liaok8s:/home/mainte/coredns# kubectl explain ct.spec.replicas
KIND:     CronTab
VERSION:  stable.example.com/v1

FIELD:    replicas <integer>

DESCRIPTION:
     副本数
```
## 创建crd实例
创建一个crontab的实例
```cpp
apiVersion: "stable.example.com/v1"
kind: CronTab
metadata:
  name: my-new-cron-object
spec:
  cronSpec: "* * * * */5"
  image: my-awesome-cron-image
  replicas: 2
```
创建完成后查看该实例，因为是一个自定义资源，没有任何的逻辑。
```cpp
crontab.stable.example.com/my-new-cron-object created
root@liaok8s:/home/mainte/coredns# kubectl get ct
NAME                 AGE
my-new-cron-object   50s
root@liaok8s:/home/mainte/coredns# kubectl get ct
NAME                 AGE
my-new-cron-object   64s
root@liaok8s:/home/mainte/coredns# kubectl get ct -o wide
NAME                 AGE
my-new-cron-object   85s
```
## 添加额外的打印列
从Kubernetes 1.11开始，kubectl使用服务器端打印。服务器决定由kubectl get命令显示哪些列即在我们获取一个内置资源的时候会显示出一些列表信息(比如：kubectl get nodes)。这里我们可以使用CustomResourceDefinition自定义这些列，当我们在查看自定义资源信息的时候显示出我们需要的列表信息。通过在crd文件中添加“additionalPrinterColumns:”字段，在该字段下声明需要打印列的的信息。
spec.versions新增

```cpp
spec:
  # 用于REST API的组名称: /apis/<group>/<version>
  group: stable.example.com
  # 此CustomResourceDefinition支持的版本列表
  versions:
    - name: v1
      # 每个版本都可以通过服务标志启用/禁用。
      served: true
      # 必须将一个且只有一个版本标记为存储版本。
      storage: true
      #使用v3定义创建容器的属性 cronSpec和image是字符串类型replicas是int型
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                cronSpec:
                  type: string
                  description: "定时任务触发时间"
                image:
                  type: string
                  description: "镜像"
                replicas:
                  type: integer
                  description: "副本数"
      additionalPrinterColumns:
        - name: cronSpec
          type: string
          description: 定时任务触发时间
          jsonPath: .spec.cronSpec
        - name: replicas
          type: integer
          description: 副本数
          jsonPath: .spec.replicas
        - name: namespace
          type: string
          description: 命名空间
          jsonPath: .metadata.namespace
```
kubectl apply后查看该定义实例
```cpp
root@liaok8s:/home/mainte/coredns# kubectl get ct
NAME                 CRONSPEC      REPLICAS   NAMESPACE
my-new-cron-object   * * * * */5   2          default
```
也可以通过获取实际对象yaml指定jsonPath
```cpp
root@liaok8s:/home/mainte/coredns# kubectl get ct -o yaml
apiVersion: v1
items:
- apiVersion: stable.example.com/v1
  kind: CronTab
  metadata:
    annotations:
      kubectl.kubernetes.io/last-applied-configuration: |
        {"apiVersion":"stable.example.com/v1","kind":"CronTab","metadata":{"annotations":{},"name":"my-new-cron-object","namespace":"default"},"spec":{"cronSpec":"* * * * */5","image":"my-awesome-cron-image","replicas":2}}
    creationTimestamp: "2022-02-17T03:21:32Z"  #比如可以指定metadata.creationTimestamp表示创建时间
    generation: 1
    name: my-new-cron-object
    namespace: default  #比如指定jsonPath=metadata.namespace就是命名空间
    resourceVersion: "314285"
    uid: d402bffe-3c04-424a-b7f9-0c4e50b4dd46
  spec:
    cronSpec: '* * * * */5'
    image: my-awesome-cron-image
    replicas: 2
kind: List
metadata:
  resourceVersion: ""
  selfLink: ""
```
## 自定义资源验证
validation这个验证是为了在创建好自定义资源后，通过该资源创建对象的时候，对象的字段中存在无效值，则创建该对象的请求将被拒绝，否则会被创建。我们可以在crd文件中添加“validation:”字段来添加相应的验证机制。

我们可以通过使用OpenAPI v3 模式来验证我们自定义的资源对象，使用该模式也应用了一些限制：

- default, nullable, discriminator, readOnly, writeOnly, xml, deprecated和$ref不能设置这些字段。
- 字段uniqueItem不能设置为true
- 字段additionalProperties不能设置为false
具体支持格式可使用explain打印
```cpp
 openAPIV3Schema     <Object>
               $ref     <string>
               $schema  <string>
               additionalItems  <>
               additionalProperties     <>
               allOf    <[]Object>
               anyOf    <[]Object>
               default  <>
               definitions      <map[string]Object>
               dependencies     <map[string]>
               description      <string>
               enum     <[]>
               example  <>
               exclusiveMaximum <boolean>
               exclusiveMinimum <boolean>
               externalDocs     <Object>
                  description   <string>
                  url   <string>
               format   <string>
               id       <string>
               items    <>
               maxItems <integer>
               maxLength        <integer>
               maxProperties    <integer>
               maximum  <number>
               minItems <integer>
               minLength        <integer>
               minProperties    <integer>
               minimum  <number>
               multipleOf       <number>
               not      <Object>
               nullable <boolean>
               oneOf    <[]Object>
               pattern  <string>
               patternProperties        <map[string]Object>
               properties       <map[string]Object>
               required <[]string>
               title    <string>
               type     <string>
               uniqueItems      <boolean>
               x-kubernetes-embedded-resource   <boolean>
               x-kubernetes-int-or-string       <boolean>
               x-kubernetes-list-map-keys       <[]string>
               x-kubernetes-list-type   <string>
               x-kubernetes-map-type    <string>
               x-kubernetes-preserve-unknown-fields     <boolean>
               x-kubernetes-validations <[]Object>
                  message       <string>
                  rule  <string>
```
修改 crd文件测试数字和正则表达式

```cpp
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                cronSpec:
                  type: string
                  description: "定时任务触发时间"
                  pattern: '^(\d+|\*)(/\d+)?(\s+(\d+|\*)(/\d+)?){4}$'
                image:
                  type: string
                  description: "镜像"
                replicas:
                  type: integer
                  description: "副本数"
                  minimum: 1
                  maximum: 10
```
修改crontab实例

```cpp
apiVersion: "stable.example.com/v1"
kind: CronTab
metadata:
  name: my-new-cron-object
spec:
  cronSpec: "* * * * */5"
  image: my-awesome-cron-image
  replicas: 12
```
当relica大小不在1-10之间是创建会出现。

```cpp
root@liaok8s:/home/mainte/coredns# kubectl apply -f crontab.yaml
The CronTab "my-new-cron-object" is invalid: spec.replicas: Invalid value: 12: spec.replicas in body should be less than or equal to 10
```
## 添加状态和伸缩配置
直接创建的my-new-cron-object 进行scale操作
```cpp
root@liaok8s:/home/mainte/coredns# kubectl scale --replicas=5 ct/my-new-cron-object
Error from server (NotFound): crontabs.stable.example.com "my-new-cron-object" not found
```
修改crd配置（新增自定义子资源）

```cpp
spec:
  # 用于REST API的组名称: /apis/<group>/<version>
  group: stable.example.com
  # 此CustomResourceDefinition支持的版本列表
  versions:
    - name: v1
      # 每个版本都可以通过服务标志启用/禁用。
      served: true
      # 必须将一个且只有一个版本标记为存储版本。
      storage: true
      #使用v3定义创建容器的属性 cronSpec和image是字符串类型replicas是int型
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                cronSpec:
                  type: string
                  description: "定时任务触发时间"
                  pattern: '^(\d+|\*)(/\d+)?(\s+(\d+|\*)(/\d+)?){4}$'
                image:
                  type: string
                  description: "镜像"
                replicas:
                  type: integer
                  description: "副本数"
                  minimum: 1
                  maximum: 10
      additionalPrinterColumns:
        - name: cronSpec
          type: string
          description: 定时任务触发时间
          jsonPath: .spec.cronSpec
        - name: replicas
          type: integer
          description: 副本数
          jsonPath: .spec.replicas
        - name: namespace
          type: string
          description: 命名空间
          jsonPath: .metadata.namespace         
     # 自定义资源的子资源的描述
      subresources:
        # 启用状态子资源。
        status: {}
        # 启用scale子资源
        scale:
          specReplicasPath: .spec.replicas #表示scale是spec.relicas字段
          statusReplicasPath: .status.replicas 
          labelSelectorPath: .status.labelSelector
```

apply之后添加后

```cpp
root@liaok8s:/home/mainte/coredns# kubectl scale --replicas=5 ct/my-new-cron-object      
crontab.stable.example.com/my-new-cron-object scaled
root@liaok8s:/home/mainte/coredns# kubectl get ct
NAME                 CRONSPEC      REPLICAS   NAMESPACE
my-new-cron-object   * * * * */5   5          default
root@liaok8s:/home/mainte/coredns# kubectl get ct -o yaml
```
