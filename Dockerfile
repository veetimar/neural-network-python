FROM alpine

RUN apk add python3 py3-numpy py3-matplotlib

WORKDIR /workdir

COPY . .

CMD ["python", "src/digits.py"]
