FROM python:3.7.7

# If STATIC_INDEX is 1, serve / with /static/index.html directly (or the static URL configured)
# ENV STATIC_INDEX 1

RUN apt update
RUN apt install -y nginx supervisor
RUN pip3 install gunicorn
RUN pip3 install setuptools
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y


COPY ./ /app
COPY requirements.txt /

RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -U -r /requirements.txt
WORKDIR /home/ubuntu/myproject/pressure_sore_api/myapp

