FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    libreoffice-java-common \
    fonts-nanum \
    fonts-noto-cjk \
    fonts-liberation \
    fontconfig \
    locales \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i '/ko_KR.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG=ko_KR.UTF-8
ENV LANGUAGE=ko_KR:ko
ENV LC_ALL=ko_KR.UTF-8

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$JAVA_HOME/bin:/usr/lib/libreoffice/program:$PATH

COPY H2Orestart.oxt /tmp/H2Orestart.oxt
RUN libreoffice --headless --nologo --nofirststartwizard --accept="socket,host=localhost,port=2002;urp;" & \
    sleep 20 && \
    unopkg add --shared /tmp/H2Orestart.oxt && \
    pkill -f soffice.bin || true && \
    rm -f /tmp/H2Orestart.oxt

RUN fc-cache -fv

WORKDIR /data

CMD ["tail", "-f", "/dev/null"]
