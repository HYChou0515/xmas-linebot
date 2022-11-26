FROM hychousvm/mrcnn:3deaec5d902d16e1daf56b62d5971d428dc920bc
RUN pip install fastapi uvicorn[standard] 
ADD . /app
RUN mv Mask_RCNN/samples/coco/ /app
WORKDIR /app
ENV DEBUG=1 \
    PYTHONUNBUFFERED=1 \
    LOGLEVEL=DEBUG
CMD ["uvicorn", "--host", "0.0.0.0", "--reload", "main:app"]
