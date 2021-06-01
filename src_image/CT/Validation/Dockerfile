FROM python:3.8-slim
WORKDIR /app
COPY requirement.txt .
RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1
RUN pip install -r requirement.txt
COPY . /app
RUN dvc import https://github.com/ImadSidhoum/BDS_pipeline_reborn src_image/data -o app/data
ENTRYPOINT ["python", "main.py"]