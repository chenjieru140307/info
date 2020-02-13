# 应用实战篇（部署验证)

### 1、创建应用

> **备注：**​ 以Python Flask为例

**a) 环境准备：**​ CentOS7.+​ Python 2.7.0+​ Flask 0.11 +

**b）环境安装：**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-4_19-58-23.png)

**c）创建应用**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_17-53-0.png)

**d）添加代码：**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_17-47-26.png)

**e）启动测试:**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-4_20-4-25.png)

**最终实现，打开浏览器输入 http://localhost:5000/**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-4_20-5-48.png)

### 2、容器化

**a）编写Dockerfile**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_17-48-18.png)

**b）构建镜像**

docker build -f dockerfile -t webapp .

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_14-50-11.png)

**c）查看本地镜像列表**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_14-51-40.png)

**d）启动构建后的容器**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_14-53-0.png)

**最终实现，打开浏览器输入 http://localhost:5000/, 容器化成功，此时你可以把该容像迁移至任何docker环境运行。**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_14-53-32.png)

### 3、添加镜像到镜像仓库

**a）创建项目**

**b）上传镜像**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-18-56.png)

### 4、代码托管

**a）登录已经安装好的Gitlab**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_15-0-29.png)

**b）创建项目**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_15-2-0.png)

**c）关联本地代码** ![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_15-4-4.png)

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_15-5-33.png)

**d）确认Gitlab仓库代码已提交**

> 刷新浏览器， 选择 Repository → 文件

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_15-7-3.png)

### 5、CI配置（CI流水线）

**a）创建gitlab CI配置文件**

> **注意：** 此文件属于隐藏文件， 在当前目录可以通过 ls -a命令查看。

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_15-16-12.png)

**b）Gitlab 配置runner, 开启共享runner**

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_15-21-33.png)

**c）重新提交代码到gitlab仓库**

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_15-20-5.png)

**d）Gitlab CI自动构建镜像**

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_15-23-46.png)

**最终，整个过程自动执行， 最终产生可以发布的镜像**

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-11-38.png)

### 6、一键发布

**a）创建应用**

> 在浏览器打开Rancher容器管理平台

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-14-50.png)

**b）添加服务**

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-31-5.png)

**c）发布验证**

> 进入webapp应用→选择web服务→选择端口→点击主机IP，此时自动跳转到我们开发的应用程序WEB页面

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-35-21.png)

最终，恭喜你， 你已经成功进入新的领域， 祝你一路顺风。

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-36-40.png)

### 7、应用升级

**a）更新代码**

> 我们继续开发webapp应用程序, 添加新代码进入app.py

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-39-56.png)

**b）提交代码**

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-41-11.png)

**c）自动流程线CI**

> 自动执行编译构建

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-42-9.png)

**产生新的docker镜像**

![1](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-43-49.png)

**d）升级发布**

> 点击升级按钮

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-44-58.png)

**点击升级， 升级过程自动下载新镜像并运行， 此时旧版本还存在**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-46-9.png)

**点击升级完成。 如果升级失败可以选择旧版本回滚**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-48-29.png)

**e）升级验证**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-55-43.png)

**最终，恭喜， 你成功了！**

![img](https://library.prof.wang/handbook/h12-opsenv/service-36/image2018-11-6_16-56-8.png)