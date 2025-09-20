# 模板
博客使用[hugo](https://themes.gohugo.io/)，主题使用[relearn](https://themes.gohugo.io/themes/hugo-theme-relearn/)。

# 安装教程

## 安装hugo
```shell
brew install hugo
choco install hugo
```

## 本地调试
```shell
  hugo server
```


# 新增markdown
在`content`目录下新增markdown文件，文件名格式自定义，每个目录下有一个_index.md文件，用于定义父导航栏标题，权重等。

## 提取图片
从csdn这些博客网站上copy下来后图片都是在线图片，有跨域问题，执行markdown.py自动将content目录下(包含子级目录)下md带有http和https得图片提取到static/images目录下，并替换markdown路径。

## 其他配置
hugo.toml配置你得baseurl,我是将项目发布到docs目录下，所以配置为
```
publishDir = "docs"
baseURL = 'https://jiaozi789.github.io/docs'
```
如果是其他github账号，修改index.html跳转即可。
# 发布
首先打包静态网页
```shell
  hugo
```
使用git push提交到github，然后使用github action自动发布到gitpage。
访问yourname.github.io即可


