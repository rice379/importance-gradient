# 第一次上传到 GitHub 的超详细步骤

下面以仓库名 `importance-gradient` 为例。

## 方法一：网页上传，最简单

适合第一次使用 GitHub、不想安装 Git 的情况。

1. 打开浏览器，进入你的 GitHub 主页：`https://github.com/rice379`
2. 登录 GitHub。
3. 右上角点 `+`。
4. 点 `New repository`。
5. `Repository name` 填：

   ```text
   importance-gradient
   ```

6. `Description` 可以填：

   ```text
   Importance-aware sparse-gradient synchronization for LLM post-training
   ```

7. 选择 `Public` 或 `Private`。
   - 想让导师/别人直接看到：选 `Public`
   - 现在还不想公开：选 `Private`
8. 不要勾选 `Add a README file`，因为本目录里已经有 README。
9. 点绿色按钮 `Create repository`。
10. 进入新仓库页面后，点 `uploading an existing file`。
11. 打开本机整理好的 `importance-gradient` 目录。

    如果你是从 Codex 生成的交付目录上传，请以 Codex 最终回复里给出的路径为准。

12. 选中里面所有文件和文件夹，拖到 GitHub 上传区域。
13. 等上传完成。
14. 页面下方 `Commit changes` 处填写：

    ```text
    Initial release of ImportanceGradient artifact
    ```

15. 点绿色按钮 `Commit changes`。

完成后，你的代码地址就是：

```text
https://github.com/rice379/importance-gradient
```

## 方法二：用 Git 命令上传，更专业

如果你的电脑还没有 Git，先安装 Git for Windows：

```text
https://git-scm.com/download/win
```

安装完以后，打开 PowerShell。

### 1. 进入整理好的代码目录

```powershell
cd 你的\importance-gradient\目录
```

### 2. 初始化 Git 仓库

```powershell
git init
```

### 3. 设置你的 GitHub 用户名和邮箱

只需要设置一次。邮箱可以用 GitHub 绑定邮箱。

```powershell
git config --global user.name "rice379"
git config --global user.email "你的邮箱@example.com"
```

### 4. 查看有哪些文件会被上传

```powershell
git status
```

### 5. 添加全部文件

```powershell
git add .
```

### 6. 提交

```powershell
git commit -m "Initial release of ImportanceGradient artifact"
```

### 7. 在 GitHub 网页创建空仓库

和方法一的第 1 到第 9 步一样。仓库名建议：

```text
importance-gradient
```

注意：不要勾选 `Add a README file`。

### 8. 绑定远程仓库

```powershell
git branch -M main
git remote add origin https://github.com/rice379/importance-gradient.git
```

### 9. 推送到 GitHub

```powershell
git push -u origin main
```

如果弹出登录窗口，就用 GitHub 账号登录。

## 上传后检查

打开：

```text
https://github.com/rice379/importance-gradient
```

确认能看到：

- `README.md`
- `importance_gradient/`
- `experiments/`
- `docs/`
- `tests/`

如果这些都在，说明上传成功。

## 常见问题

### 1. GitHub 提示文件太大

说明你不小心上传了模型、数据集、checkpoint 或日志。删除这些文件，只保留代码和少量 summary。

### 2. `git` 命令找不到

说明没有安装 Git，或者安装后没有重启 PowerShell。安装 Git for Windows 后重新打开 PowerShell。

### 3. `git push` 要求密码

GitHub 现在通常不支持直接输入账号密码。推荐用弹出的浏览器登录。如果必须输入 token，需要到 GitHub 创建 Personal Access Token。

### 4. 传错了怎么办

如果刚上传不久，可以在 GitHub 仓库页面点 `Settings` -> 最下面 `Delete this repository` 删除重建。删除仓库要谨慎。
