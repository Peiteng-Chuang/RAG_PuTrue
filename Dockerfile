FROM python:3.12-bookworm

# 安裝 Java 與系統工具
RUN apt-get update && apt-get install -y openjdk-17-jdk-headless && apt-get clean

# 設定環境變數
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# 設定工作目錄，但不 COPY 程式碼，也不寫 CMD
WORKDIR /app

# 只安裝依賴，這樣依賴就會被快取（Cache）住
COPY requirements.txt .
RUN pip install -r requirements.txt

# 這裡不寫 COPY . .，也不寫 CMD