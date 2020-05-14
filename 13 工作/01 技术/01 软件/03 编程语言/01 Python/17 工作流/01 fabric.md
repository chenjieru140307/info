
# fabric

作用：

- 使用 Fabric 来编写和执行 Python 函数或 task ，以实现与远程服务器的自动化交互。

文档：

- [fabric 中文文档](https://fabric-chs.readthedocs.io/zh_CN/chs/tutorial.html)

**举例1：**（自动部署）


```py
from fabric.api import env,run
from fabric.operations import sudo
# 这个是对 fabfile.py 的备份。
GIT_REPO='https://github.com/evo-li/blogproject.git'
env.user='xxxxx'# 填写自己的用户名
env.password='xxxxx'# 填写自己的密码

# 填写自己的主机对应的域名
env.hosts=['iterate.site']
# 一般情况下为 22 端口，如果非 22 端口请查看你的主机服务提供商提供的信息
env.port = '22'


def deploy():
    source_folder = '/home/evo/sites/iterate.site/blogproject'

    run('cd %s && git pull' % source_folder)
    run("""
        cd {} &&
        ../_env3.6/bin/pip install -r requirements.txt &&
        ../_env3.6/bin/Python3 manage.py collectstatic --noinput &&
        ../_env3.6/bin/Python3 manage.py migrate
        """.format(source_folder))
    run("""
    cd {} &&
    ../_env3.6/bin/gunicorn blogproject.wsgi:application -c deployfiles/gunicorn.conf.py
    """.format(source_folder))
    sudo('systemctl restart nginx')
```


