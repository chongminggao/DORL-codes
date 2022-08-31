FROM continuumio/miniconda3
EXPOSE 22

ARG ROOT_PASSWORD=laogaozhengao
ENV HOME=/root

LABEL MAINTAINER="@chongminggao"

RUN apt-get update && apt-get install -y openssh-server

RUN set -ex && \
    apt update && \
    apt install -y zsh openssh-server htop vim curl autossh && \
    echo root:${ROOT_PASSWORD} | chpasswd && \
    sed -i 's/#\?PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    service ssh start && \
    yes Y | sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)" && \
    sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="ys"/' $HOME/.zshrc && \
    cd ~ && wget "https://linux.chongminggao.top/configs/.gcm_bash_config?$(date +%s)" -O .gcm_bash_config && \
    echo "\n. \$HOME/.gcm_bash_config" >> .bashrc && echo "\n. \$HOME/.gcm_bash_config" >> .zshrc && \
    cd ~ && echo "exec zsh\n" >> .bash_profile && echo "exec zsh\n" >> .profile

# or ADD (difference: Add can unzip things)
COPY Dockerfile $HOME/

# or RUN (difference: CMD will not run if some user indicated command is appended to docker run)
ENTRYPOINT service ssh restart && zsh

# docker build -t chongminggao/miniconda3-zsh:v0 .