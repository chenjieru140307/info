
# 确保 Docker 已经就绪

首先，我们会查看 Docker 是否能正常工作，然后学习基本的Docker的工作流：创建并管理容器。我们将浏览容器的典型生命周期：从创建、管理到停止，直到最终删除。


## 查看 Docker 功能是否正常

```sh
$ sudo docker info

Containers: 0
 Running: 0
 Paused: 0
 Stopped: 0
Images: 0
Server Version: 17.05.0-ce
Storage Driver: aufs
 Root Dir: /var/lib/docker/aufs
 Backing Filesystem: extfs
 Dirs: 0
 Dirperm1 Supported: true
Logging Driver: json-file
Cgroup Driver: cgroupfs
Plugins: 
 Volume: local
 Network: bridge host macvlan null overlay
Swarm: inactive
Runtimes: runc
Default Runtime: runc
Init Binary: docker-init
containerd version: 9048e5e50717ea4497b757314bad98ea3763c145
runc version: 9c2d8d184e5da67c95d601382adf14862e4f2228
init version: 949e6fa
Security Options:
 apparmor
 seccomp
  Profile: default
Kernel Version: 4.4.0-117-generic
Operating System: Ubuntu 16.04.5 LTS
OSType: linux
Architecture: x86_64
CPUs: 1
Total Memory: 1.953GiB
Name: iZuf66eabunrloh2og4jgsZ
ID: VK6M:67FI:HXPJ:NEOH:YGXF:AY2C:SIHS:Y5ND:PKP6:E2T4:7R3P:JCHG
Docker Root Dir: /var/lib/docker
Debug Mode (client): false
Debug Mode (server): false
Registry: https://index.docker.io/v1/
Experimental: false
Insecure Registries:
 127.0.0.0/8
Live Restore Enabled: false

WARNING: No swap limit support
```

说明：

- docker info 可以返回：
  - 所有容器和镜像（镜像即是Docker用来构建容器的“构建块”）的数量、
  - Docker使用的执行驱动和存储驱动（execution and storage driver），
  - 以及Docker的基本配置。
