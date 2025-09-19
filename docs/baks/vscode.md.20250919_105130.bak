---
title: "vscode插件开发教程"
date: 2025-09-18T16:55:17+08:00
# bookComments: false
# bookSearchExclude: false
---

# VS Code 插件开发教程

## 概述
Visual Studio Code（简称 VS Code）是一款由 Microsoft 开发的开源轻量级编辑器，支持跨平台（Windows、macOS、Linux）。  
其最大的优势之一是**强大的插件系统**，开发者可以通过编写扩展（Extension）来增强 VS Code 的功能，比如支持新的编程语言、代码提示、调试器、界面主题等。

VS Code 插件的主要原理是：
- 插件运行在独立的进程（Extension Host）中，不会阻塞编辑器主线程。
- 插件通过 **VS Code 提供的 API** 与编辑器进行交互，比如注册命令、添加菜单、修改编辑器行为等。
- 插件开发语言主要是 **TypeScript** 或 **JavaScript**，并基于 Node.js 生态。

---

## 安装

### VS Code 安装
1. 打开 [VS Code 官方下载页面](https://code.visualstudio.com/Download)。
2. 选择对应操作系统（Windows、macOS 或 Linux）。
3. 按提示进行安装，安装完成后可以通过 `code` 命令（需要在安装时勾选“添加到 PATH”）在命令行中启动 VS Code。

### 插件开发环境安装
插件开发需要以下工具：
- **yo**（Yeoman 脚手架工具）
- **generator-code**（VS Code 插件项目生成器）
- **vsce**（VS Code Extension CLI，用于打包和发布插件）

安装步骤：
```bash
# 安装 yo 和 generator-code
npm install -g yo generator-code

# 安装 vsce
npm install -g @vscode/vsce
```
## 开发
### 生成代码

使用 Yeoman 脚手架生成插件项目：

```cmd
yo code
```
执行后会有交互式提示，例如：

-   选择插件类型（TypeScript / JavaScript）
-   插件名称
-   描述
-   初始化 Git 仓库等
    

生成完成后，项目目录大致结构如下：

```
my-extension/
├── .vscode/           # VS Code 调试配置
├── src/               # 插件源码
│   └── extension.ts   # 插件入口文件
├── package.json       # 插件描述文件，配置命令、激活事件、依赖等
├── tsconfig.json      # TypeScript 配置（如果是 TS 项目）
└── README.md          # 插件说明文档

```
- package.json：插件的核心配置文件，用来描述插件元信息和扩展点。
- extension.ts：插件入口文件，负责注册命令和功能。

#### package.json 核心配置
package.json 是插件的描述文件，控制插件如何被 VS Code 加载。主要字段：

```
{
  "name": "my-extension",
  "displayName": "My Extension",
  "description": "一个简单的 VS Code 插件示例",
  "version": "0.0.1",
  "publisher": "your-name",
  "engines": {
    "vscode": "^1.80.0"
  },
  "activationEvents": [
    "onCommand:extension.helloWorld"
  ],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "extension.helloWorld",
        "title": "Hello World"
      }
    ]
  },
  "scripts": {
    "vscode:prepublish": "npm run compile",
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./",
    "test": "npm run compile && node ./out/test/runTest.js"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "vscode": "^1.1.37"
  }
}

```
核心字段说明：

-   **name**：插件的唯一 ID（发布后不可更改）。
-   **displayName**：VS Code Marketplace 上显示的名称。
-   **version**：插件版本。
-   **publisher**：发布者名称（需与 Marketplace 发布者一致）。
-   **engines.vscode**：兼容的 VS Code 版本范围。
-   **activationEvents**：触发插件激活的事件（如 `onCommand`、`onLanguage`、`*`）。
-   **main**：插件的入口文件（一般是编译后的 `extension.js`）。
-   **contributes**：插件扩展点，例如命令、菜单、快捷键、配置等。

#### extension.ts 核心函数
extension.ts 是插件的入口文件，负责插件的生命周期和功能实现。

```
import * as vscode from 'vscode';

/**
 * 插件被激活时调用
 * @param context 插件上下文对象，包含订阅、全局存储等
 */
export function activate(context: vscode.ExtensionContext) {
    console.log('插件已激活！');

    // 注册命令
    let disposable = vscode.commands.registerCommand('extension.helloWorld', () => {
        vscode.window.showInformationMessage('Hello World from My Extension!');
    });

    // 将命令注册到插件上下文，确保插件卸载时清理资源
    context.subscriptions.push(disposable);
}

/**
 * 插件被停用时调用
 * 通常用于清理资源、保存数据
 */
export function deactivate() {}

```
核心点解释：

-   **activate**：插件激活时执行（如首次运行命令、打开特定文件类型）。
-   **deactivate**：插件停用时执行，用于清理资源。
-   **vscode.commands.registerCommand**：注册一个命令（命令 ID 必须和 `package.json` 中一致）。
-   **vscode.window.showInformationMessage**：在 VS Code 界面右下角弹出提示消息。
-   **context.subscriptions**：插件上下文，保存所有注册的资源，确保在插件停用时能正确释放。


### Hello World 示例

1.  编辑 `src/extension.ts`，添加一个最简单的命令：

```
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    console.log('插件已激活！');

    let disposable = vscode.commands.registerCommand('extension.helloWorld', () => {
        vscode.window.showInformationMessage('Hello World from My Extension!');
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}

```
2. 在 `package.json` 中配置命令：

```
{
  "contributes": {
    "commands": [
      {
        "command": "extension.helloWorld",
        "title": "Hello World"
      }
    ]
  }
}

```
3. 运行调试：

-   按 `F5` 启动调试，会打开一个新的 VS Code 窗口（Extension Development Host）。
-   打开命令面板（`Ctrl+Shift+P` / `Cmd+Shift+P`），输入并运行 **Hello World**。
-   会弹出消息 "Hello World from My Extension!"。

### 拓展介绍

VS Code 插件 API 非常丰富，常见扩展能力包括：

-   **编辑器扩展**：代码高亮、自动补全、格式化器。
    
-   **UI 扩展**：状态栏、活动栏、侧边栏视图。
    
-   **调试扩展**：调试适配器，用于支持新的调试语言。
    
-   **文件系统扩展**：实现虚拟文件系统。
    

常见配置示例（在 `package.json` 中添加）：

#### 1. 命令（Commands）
命令是最常见的扩展方式，用户可以在命令面板（Ctrl+Shift+P）或绑定快捷键来触发。

**配置（package.json）**：
```json
{
  "contributes": {
    "commands": [
      {
        "command": "extension.helloWorld",
        "title": "Hello World"
      }
    ]
  }
}
```
实现（extension.ts）：

```
vscode.commands.registerCommand('extension.helloWorld', () => {
    vscode.window.showInformationMessage('Hello World!');
});
```
#### 2. 菜单（Menus）

可以把命令挂载到编辑器右键菜单、资源管理器右键菜单等位置。

**配置（package.json）**：

```
{
  "contributes": {
  "commands": [
  {
    "command": "extension.helloWorld",
    "title": "hello"
  }，
    "menus": {
      "editor/context": [
        {
          "command": "extension.helloWorld",
          "when": "editorLangId == javascript",
          "group": "navigation"
        }
      ]
    }
  }
}

```
说明：

-   `editor/context` 表示编辑器内右键菜单。
-   `when` 条件限制了命令只在 JavaScript 文件中出现。
-   `group` 决定菜单项分组（navigation = 导航相关）。
-   菜单本身没有名字，只能通过命令 title 来显示，菜单本省command会关联到commands的命令通过command的title显示菜单名称。


菜单位置由 `menus` 的 key 决定，比如：
```
菜单位置 key:
`editor/context` 编辑器右键菜单
`editor/title` 编辑器标题栏按钮
`editor/title/context` 编辑器标题栏右键菜单
`explorer/context` 资源管理器右键菜单
`commandPalette` 命令面板（Ctrl+Shift+P）
`view/title` 视图面板标题栏按钮
`scm/title` 版本控制标题栏按钮
```
#### 3\. 快捷键（Keybindings）

可以为命令绑定快捷键。

**配置（package.json）**：
```
{
  "contributes": {
    "keybindings": [
      {
        "command": "extension.helloWorld",
        "key": "ctrl+alt+h",
        "when": "editorTextFocus"
      }
    ]
  }
}

```
说明：
-   `key`：快捷键组合。
-   `when`：触发条件，这里是“编辑器有焦点时”。

#### 4\. 状态栏（Status Bar Items）

可以在底部状态栏添加一个按钮。

**实现（extension.ts）**：
```
let statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
statusBar.text = "$(smiley) Hello";
statusBar.command = "extension.helloWorld";
statusBar.show();
context.subscriptions.push(statusBar);
```
说明：

-   `createStatusBarItem` 用于创建状态栏元素。
-   `text` 可以包含图标（如 `$(smiley)`）。
-   `command` 绑定点击事件。

#### 5\. 侧边栏视图（Views）

可以在活动栏（左侧竖栏）添加一个自定义视图。

**配置（package.json）**：

```
{
  "contributes": {
    "views": {
      "explorer": [
        {
          "id": "mySidebar",
          "name": "My Sidebar"
        }
      ]
    }
  }
}

```
实现（extension.ts）：

```
class MyTreeDataProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
    getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
        return element;
    }
    getChildren(): vscode.TreeItem[] {
        return [
            new vscode.TreeItem("Item 1"),
            new vscode.TreeItem("Item 2")
        ];
    }
}

vscode.window.registerTreeDataProvider("mySidebar", new MyTreeDataProvider());

```
说明：

-   在 **资源管理器面板** 添加一个新视图 “My Sidebar”。
    
-   用 `TreeDataProvider` 动态提供数据。
    

* * *
#### 6\. 编辑器装饰（Decorations）

可以给代码添加背景色、高亮、提示信息等。

**实现（extension.ts）**：

```
const decorationType = vscode.window.createTextEditorDecorationType({
    backgroundColor: "rgba(255,0,0,0.3)"
});

const editor = vscode.window.activeTextEditor;
if (editor) {
    const range = new vscode.Range(0, 0, 0, 5);
    editor.setDecorations(decorationType, [range]);
}

```
说明：

-   `createTextEditorDecorationType` 定义样式。
-   `setDecorations` 应用到代码范围。

* * *

#### 7\. 语言支持（Language Features）

可以扩展某种语言的代码补全、悬浮提示等。

**配置（package.json）**：

```
{
  "contributes": {
    "languages": [
      {
        "id": "mylang",
        "aliases": ["MyLang"],
        "extensions": [".mlg"],
        "configuration": "./language-configuration.json"
      }
    ]
  }
}

```
实现补全（extension.ts）：

```
vscode.languages.registerCompletionItemProvider("mylang", {
    provideCompletionItems(document, position) {
        return [new vscode.CompletionItem("helloWorld", vscode.CompletionItemKind.Keyword)];
    }
});

```
说明：

-   `languages` 定义新语言（这里是 `.mlg` 后缀）。
-   `registerCompletionItemProvider` 提供自动补全。

* * *

#### 8\. 配置（Configuration）

插件可以在 VS Code 设置里增加配置项。

**配置（package.json）**：

```
{
  "contributes": {
    "configuration": {
      "title": "My Extension",
      "properties": {
        "myExtension.enableFeature": {
          "type": "boolean",
          "default": true,
          "description": "是否启用我的功能"
        },
        "myExtension.apiEndpoint": {
          "type": "string",
          "default": "https://api.example.com",
          "description": "API 接口地址"
        }
      }
    }
  }
}

```
读取配置（extension.ts）：

```
const config = vscode.workspace.getConfiguration("myExtension");
const enable = config.get("enableFeature", true);
const api = config.get("apiEndpoint", "");
```
#### 9\. 文件系统监听（File System Watcher）

可以监听文件变化事件。

**实现（extension.ts）**：
```
const watcher = vscode.workspace.createFileSystemWatcher("**/*.js");
watcher.onDidChange(uri => console.log("修改: " + uri.fsPath));
watcher.onDidCreate(uri => console.log("创建: " + uri.fsPath));
watcher.onDidDelete(uri => console.log("删除: " + uri.fsPath));

context.subscriptions.push(watcher);
```
#### 10\. 任务（Tasks）

可以让插件在 VS Code 的“任务运行器”中提供任务。

**配置（package.json）**：

```
{
  "contributes": {
    "taskDefinitions": [
      {
        "type": "myTask",
        "required": ["taskName"],
        "properties": {
          "taskName": {
            "type": "string",
            "description": "任务名称"
          }
        }
      }
    ]
  }
}

```
实现（extension.ts）：

```
vscode.tasks.registerTaskProvider("myTask", {
    provideTasks: () => {
        return [new vscode.Task(
            { type: "myTask", taskName: "sayHello" },
            vscode.TaskScope.Workspace,
            "sayHello",
            "myTask",
            new vscode.ShellExecution("echo Hello from task!")
        )];
    },
    resolveTask: () => undefined
});

```




## 发布
### 打包插件

使用 `vsce` 打包插件：

```
# 在插件项目根目录执行
vsce package

```
执行成功后，会生成一个 .vsix 文件，例如：

```
my-extension-0.0.1.vsix
```
安装插件：

```
code --install-extension my-extension-0.0.1.vsix
```
或者到vscode插件中心右侧... install from vsix选择本地文件。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c98369d1aa61415c88e54daf215a50fe.png)

### 发布到 VS Code Marketplace

1.  前往 Azure DevOps 创建 **Publisher**。
    
2.  使用 `vsce login <publisher-name>` 登录，并输入 Personal Access Token。
    
3.  发布插件：

```
vsce publish
```
或者指定版本号：

```
vsce publish minor
```
发布成功后，你的插件就会出现在 Visual Studio Marketplace 上，供所有用户下载。