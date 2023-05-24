FROM pytorch/pytorch:latest

USER root
WORKDIR /usr/src

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev libopencv-dev

RUN pip3 install --upgrade pip \
    && pip3 install poetry \
    && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /usr/src/
RUN poetry install

# TODO: can't install poetry
RUN pip install pandas

COPY ./ /usr/src/
