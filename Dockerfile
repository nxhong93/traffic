FROM ubuntu:20.04
COPY . /traffic_sign
RUN make /traffic_sign
