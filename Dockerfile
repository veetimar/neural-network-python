FROM python

RUN pip install numpy matplotlib

WORKDIR /usr/src/app

COPY . .

CMD ["python", "src/digits.py"]
